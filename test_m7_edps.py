"""
================================================================================
Unit Tests for M7 (Event-Aware Dynamic Patch Selection)
================================================================================
Covers: importance network correctness, differentiable gating, binary
inference gating, adaptive sparsity (mechanism-level, since weights are
untrained), neighbor-aware scoring, metadata preservation, dataset
integration, DataLoader integration, visualization, report generation.

Run standalone:
    python test_m7_edps.py --dataset_root path/to/dataset
================================================================================
"""

import os
import argparse
import torch
import numpy as np

from edps_config import EDPSConfig
from edps_features import build_patch_features, feature_dim
from edps_scoring import EDPSScoringNetwork, build_normalized_adjacency_matrix
from edps_module import EDPSModule
from edps_loss import sparsity_loss
from edps_diagnostics import run_collapse_diagnostic
from edps_visualization import visualize_edps
from gate_score_logger import GateScoreLogger
from patch_config import PatchConfig
from patch_adjacency import build_adjacency_map
from patch_partition import compute_patch_grid_dims
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_generator import PatchGenerator
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m7_edps_collate import build_full_pipeline_edps_dataloader


class TestResult:
    def __init__(self):
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

    def summary(self):
        n_pass = sum(1 for _, p, _ in self.results if p)
        n_total = len(self.results)
        print(f"\n{n_pass}/{n_total} tests passed.")
        return n_pass == n_total


def run_all_m7_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M7 EDPS unit tests against: {dataset_root}\n")

    # ---- Set up a real pipeline sample once, reused by several tests ----
    pipeline_ok = True
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        pre = EventPreprocessor(PreprocessingConfig(
            use_baf=False, use_hot_pixel_filter=False, use_isolated_event_filter=False,
        ))
        vcfg = VoxelGridConfig()
        pcfg = PatchConfig(patch_size=16)
        voxel_gen = VoxelGridGenerator(vcfg)
        patch_gen = PatchGenerator(pcfg, vcfg)
        sample = dataset[0]
        cleaned = pre.process(sample, dataset_root=dataset_root)
        voxel = voxel_gen.generate({
            "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
            "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
            "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
        })
        patch_result = patch_gen.generate(voxel)
        ecfg = PatchEmbeddingConfig(embedding_dim=256)
        embed_mod = PatchEmbeddingModule(
            ecfg, in_channels=vcfg.num_channels, patch_size=pcfg.patch_size,
            n_rows=patch_result.n_rows, n_cols=patch_result.n_cols, polarity_encoding=vcfg.polarity_encoding,
        )
        embeddings, embedding_meta = embed_mod(patch_result.patches, patch_result.metadata)
    except Exception as e:
        pipeline_ok = False
        tr.record("pipeline_setup", False, str(e))

    if not pipeline_ok:
        tr.summary()
        return False, tr.results

    # ---- 1. Importance network correctness (shape, determinism) ----
    try:
        edps_cfg = EDPSConfig()
        edps = EDPSModule(edps_cfg, embedding_dim=256, num_bins=vcfg.num_bins)
        edps.eval()
        with torch.no_grad():
            out1 = edps(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
            out2 = edps(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        ok = (out1.importance_scores.shape == (embeddings.shape[0],)
              and torch.allclose(out1.importance_scores, out2.importance_scores))
        tr.record("importance_network_correctness", ok,
                   f"scores.shape={tuple(out1.importance_scores.shape)}, deterministic={torch.allclose(out1.importance_scores, out2.importance_scores)}")
    except Exception as e:
        tr.record("importance_network_correctness", False, str(e))

    # ---- 2. Differentiable gating: soft gate has gradients, values in (0,1) ----
    try:
        edps_train = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=vcfg.num_bins)
        out = edps_train(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        in_range = bool((out.importance_scores > 0).all() and (out.importance_scores < 1).all())
        loss = out.soft_gated_embeddings.sum() + sparsity_loss(
            out.importance_scores, edps_train.config.sparsity_min_ratio, edps_train.config.sparsity_max_ratio
        )
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in edps_train.parameters())
        tr.record("differentiable_gating", in_range and has_grad,
                   f"in_range={in_range}, gradients_flow={has_grad}")
    except Exception as e:
        tr.record("differentiable_gating", False, str(e))

    # ---- 3. Binary inference gating: hard mask matches threshold exactly ----
    try:
        edps_eval = EDPSModule(EDPSConfig(gate_threshold=0.5, clamp_min_ratio=0.0, clamp_max_ratio=1.0),
                                embedding_dim=256, num_bins=vcfg.num_bins)
        edps_eval.eval()
        with torch.no_grad():
            out = edps_eval(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        expected_mask = out.importance_scores > 0.5
        ok = torch.equal(out.binary_mask, expected_mask)
        tr.record("binary_inference_gating", ok,
                   f"clamp_triggered={out.token_reduction_stats['clamp_triggered']} (clamp disabled via 0/1 bounds)")
    except Exception as e:
        tr.record("binary_inference_gating", False, str(e))

    # ---- 4. Adaptive sparsity mechanism (clamp is safety-only, engages correctly at both extremes) ----
    try:
        n = embeddings.shape[0]
        edps_min = EDPSModule(EDPSConfig(gate_threshold=0.999999, clamp_min_ratio=0.1, clamp_max_ratio=0.95),
                               embedding_dim=256, num_bins=vcfg.num_bins)
        edps_max = EDPSModule(EDPSConfig(gate_threshold=0.000001, clamp_min_ratio=0.05, clamp_max_ratio=0.1),
                               embedding_dim=256, num_bins=vcfg.num_bins)
        with torch.no_grad():
            out_min = edps_min(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
            out_max = edps_max(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        min_ok = (out_min.token_reduction_stats["clamp_triggered"] and out_min.token_reduction_stats["clamp_direction"] == "min"
                  and out_min.token_reduction_stats["n_retained"] == max(1, round(0.1 * n)))
        max_ok = (out_max.token_reduction_stats["clamp_triggered"] and out_max.token_reduction_stats["clamp_direction"] == "max"
                  and out_max.token_reduction_stats["n_retained"] == round(0.1 * n))
        tr.record("adaptive_sparsity_clamp_mechanism", min_ok and max_ok,
                   f"min_clamp={out_min.token_reduction_stats}, max_clamp={out_max.token_reduction_stats}")
    except Exception as e:
        tr.record("adaptive_sparsity_clamp_mechanism", False, str(e))

    # ---- 5. Neighbor-aware scoring: different adjacency -> different scores ----
    try:
        torch.manual_seed(0)
        edps_a = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=vcfg.num_bins)
        torch.manual_seed(0)
        edps_b = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=vcfg.num_bins)
        # identical weights (same seed), but feed a DIFFERENT adjacency to edps_b
        alt_adjacency = build_adjacency_map(patch_result.n_rows, patch_result.n_cols,
                                             PatchConfig(adjacency_mode="8"))
        with torch.no_grad():
            out_a = edps_a(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
            out_b = edps_b(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, alt_adjacency)
        ok = not torch.allclose(out_a.importance_scores, out_b.importance_scores)
        tr.record("neighbor_aware_scoring", ok, "4-neighborhood vs 8-neighborhood adjacency produces different scores")
    except Exception as e:
        tr.record("neighbor_aware_scoring", False, str(e))

    # ---- 6. Metadata preservation ----
    try:
        edps_meta = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=vcfg.num_bins)
        with torch.no_grad():
            out = edps_meta(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        selected_indices_from_meta = {m.patch_index for m in out.selected_metadata}
        selected_indices_from_mask = set(out.binary_mask.nonzero(as_tuple=True)[0].tolist())
        ok = (len(out.selected_metadata) == out.selected_embeddings.shape[0] == int(out.binary_mask.sum())
              and selected_indices_from_meta == selected_indices_from_mask)
        tr.record("metadata_preservation", ok,
                   f"n_selected_meta={len(out.selected_metadata)}, n_selected_embed={out.selected_embeddings.shape[0]}")
    except Exception as e:
        tr.record("metadata_preservation", False, str(e))

    # ---- 7. Dataset integration ----
    try:
        ok = embeddings.shape[0] == len(patch_result.metadata)
        tr.record("dataset_integration", ok)
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 8. DataLoader integration ----
    try:
        n_rows, n_cols, _, _ = compute_patch_grid_dims(
            dataset[0]["sensor_height"], dataset[0]["sensor_width"], PatchConfig(patch_size=16)
        )
        loader = build_full_pipeline_edps_dataloader(dataset, dataset_root, n_rows, n_cols, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        ok = len(batch) == min(3, len(dataset)) and all("edps_output" in item for item in batch)
        tr.record("dataloader_integration", ok, f"batch_len={len(batch)}")
    except Exception as e:
        tr.record("dataloader_integration", False, str(e))

    # ---- 9. Visualization ----
    try:
        edps_viz = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=vcfg.num_bins)
        with torch.no_grad():
            out = edps_viz(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        out_path = "test_m7_edps_viz.png"
        saved = visualize_edps(out.importance_scores, out.binary_mask, patch_result.metadata,
                                patch_result.n_rows, patch_result.n_cols, edps_viz.config.gate_threshold,
                                out.token_reduction_stats, out_path=out_path)
        ok = os.path.exists(saved) and os.path.getsize(saved) > 0
        tr.record("visualization", ok, f"saved to {saved}")
        if cleanup and ok:
            os.remove(saved)
    except Exception as e:
        tr.record("visualization", False, str(e))

    # ---- 10. Collapse diagnostic + gate score logger sanity ----
    try:
        densities = np.array([s.event_density for s in patch_result.stats])
        diag = run_collapse_diagnostic(out.importance_scores.detach().numpy(), densities, warn_threshold=0.9)
        logger = GateScoreLogger()
        logger.log(out.importance_scores, threshold=0.5, step=0)
        ok = "score_density_correlation" in diag and logger.summary()["num_logged_steps"] == 1
        tr.record("collapse_diagnostic_and_gate_logger", ok, str(diag))
    except Exception as e:
        tr.record("collapse_diagnostic_and_gate_logger", False, str(e))

    # ---- 11. Report generation (importability smoke test) ----
    try:
        from generate_m7_report import generate_m7_report  # noqa
        tr.record("report_generation_importable", True)
    except Exception as e:
        tr.record("report_generation_importable", False, str(e))

    all_passed = tr.summary()
    return all_passed, tr.results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--window_us", type=int, default=200_000)
    args = ap.parse_args()

    passed, _ = run_all_m7_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
