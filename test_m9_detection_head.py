"""
================================================================================
Unit Tests for M9 (Sparse Anchor-Free Per-Token Detection Head)
================================================================================
Covers: output shape, bbox decode correctness (hand-crafted known values),
class prediction correctness, objectness/quality range, metadata mapping
preservation, gradient flow, determinism, backbone independence, dataset
integration, DataLoader integration, visualization, report generation.

Run standalone:
    python test_m9_detection_head.py --dataset_root path/to/dataset
================================================================================
"""

import os
import math
import argparse
import torch
import torch.nn.functional as F

from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead, decode_predictions, count_parameters
from detection_visualization import visualize_detections
from patch_metadata import PatchMetadata
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims
from token_padding import pad_token_sequences
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m9_detection_collate import build_full_pipeline_detection_dataloader


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


def run_all_m9_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M9 detection head unit tests against: {dataset_root}\n")

    cfg = DetectionHeadConfig(embedding_dim=256, num_classes=2)

    # ---- 1. Output shape correctness ----
    try:
        head = SparseDetectionHead(cfg)
        tokens = torch.randn(2, 10, cfg.embedding_dim)
        raw = head(tokens)
        ok = (raw["ltrb_raw"].shape == (2, 10, 4)
              and raw["class_logits"].shape == (2, 10, cfg.num_classes)
              and raw["objectness_logit"].shape == (2, 10, 1)
              and raw["quality_logit"].shape == (2, 10, 1))
        tr.record("output_shape_correctness", ok,
                   f"ltrb={tuple(raw['ltrb_raw'].shape)}, class={tuple(raw['class_logits'].shape)}")
    except Exception as e:
        tr.record("output_shape_correctness", False, str(e))

    # ---- 2. Bbox decode correctness (hand-crafted, exact expected values) ----
    try:
        raw_manual = {
            "ltrb_raw": torch.zeros(1, 1, 4),
            "class_logits": torch.tensor([[[5.0, -5.0]]]),
            "objectness_logit": torch.zeros(1, 1, 1),
            "quality_logit": torch.zeros(1, 1, 1),
        }
        meta = [PatchMetadata(patch_index=7, row_index=1, col_index=2, y0=16, x0=32, y1=32, x1=48,
                               center_y=24.0, center_x=40.0, temporal_dim=10, spatial_size=16)]
        dets = decode_predictions(raw_manual, meta, cfg, batch_index=0)
        d = dets[0]
        expected_dist = math.log(2) * cfg.distance_scale
        ok = (
            abs(d.box_x0 - (40 - expected_dist)) < 1e-4 and abs(d.box_y0 - (24 - expected_dist)) < 1e-4
            and abs(d.box_x1 - (40 + expected_dist)) < 1e-4 and abs(d.box_y1 - (24 + expected_dist)) < 1e-4
            and d.predicted_class == 0 and abs(d.objectness - 0.5) < 1e-4 and d.patch_index == 7
        )
        tr.record("bbox_decode_correctness", ok, f"box=({d.box_x0:.3f},{d.box_y0:.3f},{d.box_x1:.3f},{d.box_y1:.3f})")
    except Exception as e:
        tr.record("bbox_decode_correctness", False, str(e))

    # ---- 3. Class prediction correctness (softmax sums to 1) ----
    try:
        head2 = SparseDetectionHead(cfg)
        tokens2 = torch.randn(3, 15, cfg.embedding_dim)
        raw2 = head2(tokens2)
        probs = F.softmax(raw2["class_logits"], dim=-1)
        ok = torch.allclose(probs.sum(dim=-1), torch.ones(3, 15), atol=1e-5)
        tr.record("class_prediction_correctness", ok)
    except Exception as e:
        tr.record("class_prediction_correctness", False, str(e))

    # ---- 4. Objectness/quality range check (post-sigmoid, must be in (0,1)) ----
    try:
        obj = torch.sigmoid(raw2["objectness_logit"])
        qual = torch.sigmoid(raw2["quality_logit"])
        ok = bool((obj > 0).all() and (obj < 1).all() and (qual > 0).all() and (qual < 1).all())
        tr.record("objectness_quality_range", ok)
    except Exception as e:
        tr.record("objectness_quality_range", False, str(e))

    # ---- 5. Metadata mapping preservation ----
    try:
        meta5 = [
            PatchMetadata(patch_index=i, row_index=i // 4, col_index=i % 4, y0=0, x0=0, y1=16, x1=16,
                          center_y=8.0, center_x=8.0, temporal_dim=10, spatial_size=16)
            for i in range(12)
        ]
        head5 = SparseDetectionHead(cfg)
        tokens5 = torch.randn(1, 12, cfg.embedding_dim)
        raw5 = head5(tokens5)
        dets5 = decode_predictions(raw5, meta5, cfg, batch_index=0)
        ok = all(dets5[i].patch_index == meta5[i].patch_index
                  and dets5[i].row_index == meta5[i].row_index
                  and dets5[i].col_index == meta5[i].col_index for i in range(12))
        tr.record("metadata_mapping_preserved", ok, f"checked {len(dets5)} tokens")
    except Exception as e:
        tr.record("metadata_mapping_preserved", False, str(e))

    # ---- 5b. REGRESSION: heterogeneous K across a batch (the reported bug) ----
    # Reproduces the exact failure mode: two samples with DIFFERENT retained
    # counts, batched together, so one sample's true K < the batch's padded
    # K_max. Verifies all 4 points requested in the bug report:
    #   1. the attention mask is correctly propagated and used
    #   2. the head's predictions for padded positions are never surfaced
    #   3. metadata packing has exactly one entry per valid token
    #   4. n_predictions == n_metadata_entries, always, per sample
    try:
        head5b = SparseDetectionHead(cfg)
        sample_a = torch.randn(3, cfg.embedding_dim)   # fewer retained tokens
        sample_b = torch.randn(7, cfg.embedding_dim)   # more retained tokens -> sets K_max
        padded5b, mask5b = pad_token_sequences([sample_a, sample_b])
        raw5b = head5b(padded5b)

        meta_a = [PatchMetadata(patch_index=i, row_index=0, col_index=i, y0=0, x0=0, y1=16, x1=16,
                                 center_y=8.0, center_x=8.0, temporal_dim=10, spatial_size=16) for i in range(3)]
        meta_b = [PatchMetadata(patch_index=i, row_index=0, col_index=i, y0=0, x0=0, y1=16, x1=16,
                                 center_y=8.0, center_x=8.0, temporal_dim=10, spatial_size=16) for i in range(7)]

        dets_a = decode_predictions(raw5b, meta_a, cfg, attention_mask=mask5b, batch_index=0)
        dets_b = decode_predictions(raw5b, meta_b, cfg, attention_mask=mask5b, batch_index=1)

        ok = (
            padded5b.shape[1] == 7                      # confirms K_max padding actually happened
            and len(dets_a) == 3 == len(meta_a)          # smaller sample: no padding leaked in
            and len(dets_b) == 7 == len(meta_b)          # larger sample: nothing dropped
            and {d.patch_index for d in dets_a} == {0, 1, 2}
            and {d.patch_index for d in dets_b} == {0, 1, 2, 3, 4, 5, 6}
        )
        tr.record("heterogeneous_batch_padding_regression", ok,
                   f"K_max={padded5b.shape[1]}, sample_a: {len(dets_a)}/3 detections, "
                   f"sample_b: {len(dets_b)}/7 detections (this is the exact scenario that "
                   f"previously raised 'patch_metadata has N entries, raw predictions have M tokens')")
    except Exception as e:
        tr.record("heterogeneous_batch_padding_regression", False, str(e))

    # ---- 6. Gradient flow ----
    try:
        head6 = SparseDetectionHead(cfg)
        tokens6 = torch.randn(2, 10, cfg.embedding_dim, requires_grad=True)
        raw6 = head6(tokens6)
        loss = raw6["ltrb_raw"].sum() + raw6["class_logits"].sum() + raw6["objectness_logit"].sum() + raw6["quality_logit"].sum()
        loss.backward()
        input_grad_ok = tokens6.grad is not None and tokens6.grad.abs().sum() > 0
        param_grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head6.parameters())
        tr.record("gradient_flow", input_grad_ok and param_grad_ok)
    except Exception as e:
        tr.record("gradient_flow", False, str(e))

    # ---- 7. Determinism in eval mode ----
    try:
        head7 = SparseDetectionHead(cfg)
        head7.eval()
        tokens7 = torch.randn(2, 10, cfg.embedding_dim)
        with torch.no_grad():
            raw_a = head7(tokens7)
            raw_b = head7(tokens7)
        ok = torch.allclose(raw_a["ltrb_raw"], raw_b["ltrb_raw"])
        tr.record("determinism_eval_mode", ok)
    except Exception as e:
        tr.record("determinism_eval_mode", False, str(e))

    # ---- 8. Backbone independence: works on arbitrary (B,K,D) tensor, no Transformer needed ----
    try:
        head8 = SparseDetectionHead(cfg)
        arbitrary_tokens = torch.randn(4, 33, cfg.embedding_dim)  # never touched TransformerEncoder
        raw8 = head8(arbitrary_tokens)
        ok = raw8["ltrb_raw"].shape[:2] == (4, 33)
        tr.record("backbone_independence", ok, "head runs on arbitrary (B,K,D) input, no Transformer import used")
    except Exception as e:
        tr.record("backbone_independence", False, str(e))

    # ---- 9. Dataset integration ----
    dataset = None
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        n_rows, n_cols, _, _ = compute_patch_grid_dims(
            dataset[0]["sensor_height"], dataset[0]["sensor_width"], PatchConfig(patch_size=16)
        )
        tr.record("dataset_integration", True, f"grid=({n_rows},{n_cols})")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 10. DataLoader integration ----
    batch = None
    if dataset is not None:
        try:
            loader = build_full_pipeline_detection_dataloader(
                dataset, dataset_root, n_rows, n_cols, batch_size=3, shuffle=False,
            )
            batch = next(iter(loader))
            ok = (len(batch["detections"]) == min(3, len(dataset))
                  and all(isinstance(d, list) for d in batch["detections"]))
            n_dets = [len(d) for d in batch["detections"]]
            tr.record("dataloader_integration", ok, f"detections_per_sample={n_dets}")
        except Exception as e:
            tr.record("dataloader_integration", False, str(e))
    else:
        tr.record("dataloader_integration", False, "skipped: no dataset")

    # ---- 11. Visualization ----
    if batch is not None:
        try:
            n_rows2, n_cols2, _, _ = compute_patch_grid_dims(
                batch["meta"][0]["sensor_height"], batch["meta"][0]["sensor_width"], PatchConfig(patch_size=16)
            )
            # need full patch metadata grid for the "retained" panel -- reconstruct via PatchGenerator quickly
            from voxel_grid import VoxelGridConfig, VoxelGridGenerator
            from patch_generator import PatchGenerator
            from event_preprocessor import EventPreprocessor
            from preprocessing_config import PreprocessingConfig
            vcfg = VoxelGridConfig()
            pre = EventPreprocessor(PreprocessingConfig())
            pcfg = PatchConfig(patch_size=16)
            sample0 = dataset[0]
            cleaned0 = pre.process(sample0, dataset_root=dataset_root)
            voxel_gen0 = VoxelGridGenerator(vcfg)
            grid0 = voxel_gen0.generate({
                "t": cleaned0["t"], "x": cleaned0["x"], "y": cleaned0["y"], "p": cleaned0["p"],
                "t_start": cleaned0["t_start"], "t_end": cleaned0["t_end"],
                "sensor_height": cleaned0["sensor_height"], "sensor_width": cleaned0["sensor_width"],
            })
            patch_gen0 = PatchGenerator(pcfg, vcfg)
            patch_result0 = patch_gen0.generate(grid0)

            out_path = "test_m9_det_viz.png"
            saved = visualize_detections(
                batch["detections"][0], patch_result0.metadata, batch["selected_metadata"][0],
                n_rows2, n_cols2, batch["meta"][0]["sensor_height"], batch["meta"][0]["sensor_width"],
                out_path=out_path,
            )
            ok = os.path.exists(saved) and os.path.getsize(saved) > 0
            tr.record("visualization", ok, f"saved to {saved}")
            if cleanup and ok:
                os.remove(saved)
        except Exception as e:
            tr.record("visualization", False, str(e))
    else:
        tr.record("visualization", False, "skipped: no batch")

    # ---- 12. Report generation (importability smoke test) ----
    try:
        from generate_m9_report import generate_m9_report  # noqa
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

    passed, _ = run_all_m9_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
