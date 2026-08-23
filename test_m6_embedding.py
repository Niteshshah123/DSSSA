"""
================================================================================
Unit Tests for M6 (Patch Embedding)
================================================================================
Covers: output shape, embedding correctness, positional encoding, temporal
encoding, metadata correctness, dataset integration, DataLoader integration,
visualization, report generation.

Run standalone:
    python test_m6_embedding.py --dataset_root path/to/dataset
================================================================================
"""

import os
import argparse
import torch
import numpy as np

from patch_embedding_config import PatchEmbeddingConfig
from patch_projections import build_patch_projection
from positional_encoding import PositionalEncoding
from temporal_encoding import compute_temporal_activity, TemporalEncoding
from patch_embedding_module import PatchEmbeddingModule, count_parameters
from embedding_visualization import visualize_embeddings
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_generator import PatchGenerator
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m6_embedding_collate import build_full_pipeline_embedding_dataloader


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


def run_all_m6_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M6 patch embedding unit tests against: {dataset_root}\n")

    N, C, P, D = 20, 10, 16, 256

    # ---- 1. Output shape correctness (all three projection types) ----
    try:
        patches = torch.randn(N, C, P, P)
        ok = True
        details = []
        for ptype in ("linear", "cnn", "conv3d"):
            proj = build_patch_projection(ptype, C, P, D)
            out = proj(patches)
            details.append(f"{ptype}={tuple(out.shape)}")
            if out.shape != (N, D):
                ok = False
        tr.record("output_shape_correctness", ok, ", ".join(details))
    except Exception as e:
        tr.record("output_shape_correctness", False, str(e))

    # ---- 2. Embedding correctness: linear projection matches manual matmul ----
    try:
        proj = build_patch_projection("linear", C, P, D)
        patches_flat = patches.reshape(N, -1)
        expected = patches_flat @ proj.linear.weight.T + proj.linear.bias
        actual = proj(patches)
        ok = torch.allclose(expected, actual, atol=1e-5)
        tr.record("embedding_correctness_linear_matches_manual_matmul", ok)
    except Exception as e:
        tr.record("embedding_correctness_linear_matches_manual_matmul", False, str(e))

    # ---- 3. Positional encoding: distinct positions -> distinct codes; normalized coords correct ----
    try:
        n_rows, n_cols = 15, 19
        rows = [0, 0, 14, 7]
        cols = [0, 18, 18, 9]
        ok = True
        for mode in ("learnable", "sinusoidal"):
            pe = PositionalEncoding(D, mode, n_rows, n_cols)
            out = pe(rows, cols)
            distinct = len(torch.unique(out, dim=0)) == len(rows)
            info = pe.compute_positional_info(rows, cols)
            coords_ok = (abs(info[2].norm_row - 1.0) < 1e-9 and abs(info[2].norm_col - 1.0) < 1e-9
                         and abs(info[0].norm_row) < 1e-9 and abs(info[0].norm_col) < 1e-9)
            if not (distinct and coords_ok):
                ok = False
        tr.record("positional_encoding_correctness", ok)
    except Exception as e:
        tr.record("positional_encoding_correctness", False, str(e))

    # ---- 4. Temporal encoding: activity profile correctness + no-NaN on empty patch ----
    try:
        patches_t = torch.zeros(3, 10, P, P)
        patches_t[0, 0] = 1.0
        patches_t[1, 9] = 1.0
        # patches_t[2] stays all-zero -- edge case
        activity = compute_temporal_activity(patches_t, "signed", 10)
        ok = (torch.allclose(activity[0], torch.eye(10)[0], atol=1e-6)
              and torch.allclose(activity[1], torch.eye(10)[9], atol=1e-6)
              and not torch.isnan(activity[2]).any() and activity[2].sum().item() == 0.0)
        te = TemporalEncoding(10, D, "sinusoidal")
        out = te(activity)
        ok = ok and out.shape == (3, D) and not torch.isnan(out).any()
        tr.record("temporal_encoding_correctness", ok)
    except Exception as e:
        tr.record("temporal_encoding_correctness", False, str(e))

    # ---- 5. Metadata correctness (row/col/patch_index alignment through the full module) ----
    dataset = None
    patch_result = None
    embed_mod = None
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

        ecfg = PatchEmbeddingConfig(embedding_dim=D)
        embed_mod = PatchEmbeddingModule(
            ecfg, in_channels=vcfg.num_channels, patch_size=pcfg.patch_size,
            n_rows=patch_result.n_rows, n_cols=patch_result.n_cols, polarity_encoding=vcfg.polarity_encoding,
        )
        embeddings, meta_out = embed_mod(patch_result.patches, patch_result.metadata)

        ok = all(
            meta_out[i].patch_index == patch_result.metadata[i].patch_index
            and meta_out[i].row_index == patch_result.metadata[i].row_index
            and meta_out[i].col_index == patch_result.metadata[i].col_index
            for i in range(len(meta_out))
        )
        tr.record("metadata_correctness", ok, f"checked {len(meta_out)} entries")
    except Exception as e:
        tr.record("metadata_correctness", False, str(e))

    # ---- 6. No pruning: output count always equals input patch count ----
    try:
        ok = embeddings.shape[0] == patch_result.patches.shape[0] == len(patch_result.metadata)
        tr.record("no_pruning_every_patch_embedded", ok,
                   f"input_patches={patch_result.patches.shape[0]}, output_embeddings={embeddings.shape[0]}")
    except Exception as e:
        tr.record("no_pruning_every_patch_embedded", False, str(e))

    # ---- 7. Dataset integration ----
    try:
        ok = embeddings.shape == (len(patch_result.metadata), D)
        tr.record("dataset_integration", ok, f"embeddings.shape={tuple(embeddings.shape)}")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 8. DataLoader integration ----
    if dataset is not None:
        try:
            n_rows, n_cols, _, _ = compute_patch_grid_dims(
                dataset[0]["sensor_height"], dataset[0]["sensor_width"], PatchConfig(patch_size=16)
            )
            loader = build_full_pipeline_embedding_dataloader(
                dataset, dataset_root, n_rows, n_cols, batch_size=3, shuffle=False,
            )
            batch = next(iter(loader))
            ok = (batch["embeddings"].ndim == 3 and batch["embeddings"].shape[0] == min(3, len(dataset))
                  and batch["embeddings"].shape[2] == 256)
            tr.record("dataloader_integration", ok, f"embeddings batch shape={tuple(batch['embeddings'].shape)}")
        except Exception as e:
            tr.record("dataloader_integration", False, str(e))
    else:
        tr.record("dataloader_integration", False, "skipped: no dataset")

    # ---- 9. Visualization ----
    try:
        out_path = "test_m6_embedding_viz.png"
        activity = np.array([s.total_event_count for s in patch_result.stats])
        saved = visualize_embeddings(embeddings, out_path=out_path, activity=activity, include_tsne=True)
        ok = os.path.exists(saved) and os.path.getsize(saved) > 0
        tr.record("visualization", ok, f"saved to {saved}")
        if cleanup and ok:
            os.remove(saved)
    except Exception as e:
        tr.record("visualization", False, str(e))

    # ---- 10. Report generation (importability smoke test) ----
    try:
        from generate_m6_report import generate_m6_report  # noqa
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

    passed, _ = run_all_m6_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
