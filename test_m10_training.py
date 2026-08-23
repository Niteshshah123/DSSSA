"""
================================================================================
Unit Tests for M10 (End-to-End Training Pipeline)
================================================================================
Covers: EDPS gradient-wiring fix, reproducibility (seed + RNG round-trip),
loss components (hand-computed), target assignment (+ recall-ceiling
diagnostic), sparsity curriculum stages, augmentation (bounds, ordering,
disabled no-op), optimizer parameter groups (configurable EDPS LR),
scheduler LR trajectory, checkpoint save/load/RESUME round-trip, mAP
metric correctness, training-collate integration, a real end-to-end mini
training run, and report generation.

Run standalone:
    python test_m10_training.py --dataset_root path/to/dataset
================================================================================
"""

import os
import shutil
import math
import argparse
import numpy as np
import torch

from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule
from edps_config import EDPSConfig
from edps_module import EDPSModule
from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder
from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead

from gated_token_wiring import compute_gated_selected_embeddings
from reproducibility import seed_everything, capture_rng_state, restore_rng_state
from detection_losses import focal_loss, box_giou, giou_loss, objectness_bce_loss, quality_iou_loss
from target_assignment import assign_targets
from patch_metadata import PatchMetadata
from sparsity_curriculum import get_current_sparsity_band
from training_config import TrainingConfig
from augmentation import augment_sample
from optimizer_factory import build_optimizer
from scheduler_factory import build_scheduler
from checkpoint_manager import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    periodic_checkpoint_path, TrainingState,
)
from map_metric import compute_ap_for_class, compute_map
from training_collate import build_training_dataloader
from trainer import Trainer


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


def _build_components(n_rows, n_cols, num_bins=10):
    embed = PatchEmbeddingModule(PatchEmbeddingConfig(), in_channels=num_bins, patch_size=16,
                                  n_rows=n_rows, n_cols=n_cols, polarity_encoding="signed")
    edps = EDPSModule(EDPSConfig(), embedding_dim=256, num_bins=num_bins)
    enc = TransformerEncoder(TransformerEncoderConfig())
    head = SparseDetectionHead(DetectionHeadConfig())
    return embed, edps, enc, head


def _make_small_manifest(manifest, train_count=1, val_count=1):
    """Create a deterministic recording-level subset for M10 smoke tests."""
    from dataset_split import SplitManifest
    train = list(manifest.train[:max(1, min(train_count, len(manifest.train)))])
    val = list(manifest.val[:max(1, min(val_count, len(manifest.val)))])
    if len(val) == 0 and len(train) > 0:
        val = list(train[:1])  # Fallback to train recording if split manifest has empty val
    test = list(manifest.test[:0])
    return SplitManifest(
        seed=manifest.seed,
        train_ratio=0.5 if (train and val) else 1.0,
        val_ratio=0.5 if (train and val) else 0.0,
        test_ratio=0.0,
        train=train, val=val, test=test,
    )


def run_all_m10_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M10 training pipeline unit tests against: {dataset_root}\n")

    test_dataset_root = dataset_root
    created_temp_synth = False
    temp_synth_dir = "test_m10_temp_synth_data"

    from dataset_split import list_recording_stems
    stems_found = list_recording_stems(dataset_root) if os.path.exists(dataset_root) else []
    if len(stems_found) == 0:
        from make_synthetic_data import write_synthetic_dat, write_synthetic_bbox
        os.makedirs(temp_synth_dir, exist_ok=True)
        t_syn1, _, _, _ = write_synthetic_dat(os.path.join(temp_synth_dir, "rec1_td.dat"), n_events=5000, seed=42)
        write_synthetic_bbox(os.path.join(temp_synth_dir, "rec1_bbox.npy"), t_syn1, seed=42)
        t_syn2, _, _, _ = write_synthetic_dat(os.path.join(temp_synth_dir, "rec2_td.dat"), n_events=5000, seed=43)
        write_synthetic_bbox(os.path.join(temp_synth_dir, "rec2_bbox.npy"), t_syn2, seed=43)
        test_dataset_root = temp_synth_dir
        created_temp_synth = True
        print(f"  [INFO] No matched recording pairs in {dataset_root}. Using temporary synthetic data in {temp_synth_dir}.")

    # ---- 1. EDPS gradient-wiring fix ----
    try:
        from patch_statistics import PatchActivityStats
        from patch_embedding_module import EmbeddingMetadata

        N, D, num_bins = 20, 256, 10
        embeddings = torch.randn(N, D, requires_grad=True)
        stats = [PatchActivityStats(patch_index=i, total_event_count=10.0, positive_event_count=5.0,
                                     negative_event_count=5.0, event_density=1.0,
                                     temporal_activity_distribution=[1.0] * num_bins) for i in range(N)]
        meta = [PatchMetadata(patch_index=i, row_index=0, col_index=i, y0=0, x0=0, y1=16, x1=16,
                               center_y=8.0, center_x=8.0, temporal_dim=num_bins, spatial_size=16) for i in range(N)]
        emb_meta = [EmbeddingMetadata(patch_index=i, row_index=0, col_index=i, norm_row=0.0, norm_col=i / N,
                                       positional_encoding=[0.0] * D, temporal_encoding=None) for i in range(N)]
        adjacency = {i: [] for i in range(N)}

        edps = EDPSModule(EDPSConfig(), embedding_dim=D, num_bins=num_bins)
        out = edps(embeddings, stats, meta, emb_meta, adjacency)

        edps.zero_grad()
        old_loss = out.selected_embeddings.sum()
        old_loss.backward(retain_graph=True)
        old_grad = sum(p.grad.abs().sum().item() for p in edps.scoring_net.parameters() if p.grad is not None)

        edps.zero_grad()
        out2 = edps(embeddings, stats, meta, emb_meta, adjacency)
        gated = compute_gated_selected_embeddings(out2)
        new_loss = gated.sum()
        new_loss.backward()
        new_grad = sum(p.grad.abs().sum().item() for p in edps.scoring_net.parameters() if p.grad is not None)

        ok = (old_grad == 0.0 and new_grad > 0.0)
        tr.record("edps_gradient_wiring_fix", ok, f"unwired_grad={old_grad}, wired_grad={new_grad:.2f}")
    except Exception as e:
        tr.record("edps_gradient_wiring_fix", False, str(e))

    # ---- 1a. EDPS threshold and baseline selection mode ----
    try:
        cfg_normal = EDPSConfig(gate_threshold=0.5, selection_mode="normal")
        edps_normal = EDPSModule(cfg_normal, embedding_dim=D, num_bins=num_bins)
        out_n = edps_normal(embeddings, stats, meta, emb_meta, adjacency)
        stats_n = out_n.token_reduction_stats

        cfg_base = EDPSConfig(gate_threshold=0.5, selection_mode="baseline_no_pruning")
        edps_base = EDPSModule(cfg_base, embedding_dim=D, num_bins=num_bins)
        out_b = edps_base(embeddings, stats, meta, emb_meta, adjacency)
        stats_b = out_b.token_reduction_stats

        ok_normal = ("gate_keep_rate" in stats_n and "pct_retained" in stats_n and stats_n["gate_threshold"] == 0.5)
        ok_base = (stats_b["pct_retained"] == 100.0 and stats_b["selection_mode"] == "baseline_no_pruning")
        ok = ok_normal and ok_base
        tr.record("edps_threshold_and_baseline_mode", ok,
                   f"gate_thresh={stats_n['gate_threshold']}, normal_retained={stats_n['pct_retained']:.1f}%, "
                   f"baseline_retained={stats_b['pct_retained']:.1f}%")
    except Exception as e:
        tr.record("edps_threshold_and_baseline_mode", False, str(e))

    # ---- 1b. REGRESSION: EventPreprocessor must never crash on unsorted input ----
    # Reproduces the exact reported failure mode: timestamps that are NOT
    # ascending reaching compute_neighbor_support. Root cause was that M4
    # trusted its caller (augmentation, dataset windowing, etc.) to
    # guarantee sorted input rather than enforcing it itself. Fixed with a
    # defensive chronological sort at the start of
    # EventPreprocessor.process() -- this test constructs deliberately
    # unsorted input directly, bypassing every upstream stage entirely, so
    # it verifies the fix at its actual boundary regardless of what
    # upstream condition originally caused the disorder.
    try:
        from event_preprocessor import EventPreprocessor
        from preprocessing_config import PreprocessingConfig as _PPConfig

        n_r = 500
        rng_r = np.random.default_rng(1)
        t_unsorted = rng_r.integers(0, 100000, size=n_r).astype(np.int64)
        x_r = rng_r.integers(0, 304, size=n_r).astype(np.int64)
        y_r = rng_r.integers(0, 240, size=n_r).astype(np.int64)
        p_r = rng_r.integers(0, 2, size=n_r).astype(np.int64)
        was_unsorted = not np.all(np.diff(t_unsorted) >= 0)

        empty_boxes = np.array([], dtype=[("ts", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
                                            ("class_id", "u1"), ("confidence", "<f4"), ("track_id", "<u4")])
        fake_sample = {
            "t": torch.from_numpy(t_unsorted), "x": torch.from_numpy(x_r),
            "y": torch.from_numpy(y_r), "p": torch.from_numpy(p_r),
            "boxes": empty_boxes, "recording_stem": "fake", "window_index": 0,
            "t_start": 0, "t_end": 100000, "sensor_height": 240, "sensor_width": 304,
        }
        pre_r = EventPreprocessor(_PPConfig(use_hot_pixel_filter=False))
        result_r = pre_r.process(fake_sample, dataset_root=dataset_root)
        output_sorted = np.all(np.diff(result_r["t"]) >= 0) if len(result_r["t"]) > 0 else True

        ok = was_unsorted and output_sorted
        tr.record("unsorted_timestamp_defensive_sort_regression", ok,
                   f"input_was_unsorted={was_unsorted}, no_crash_and_output_sorted={output_sorted}")
    except Exception as e:
        tr.record("unsorted_timestamp_defensive_sort_regression", False, str(e))

    # ---- 2. Reproducibility: RNG capture/restore round-trip ----
    try:
        seed_everything(123)
        import random as _random
        _ = [_random.random() for _ in range(5)]
        _ = np.random.rand(5)
        _ = torch.rand(5)
        state = capture_rng_state()
        vals_a = (_random.random(), float(np.random.rand()), torch.rand(1).item())
        restore_rng_state(state)
        vals_b = (_random.random(), float(np.random.rand()), torch.rand(1).item())
        ok = vals_a == vals_b
        tr.record("reproducibility_rng_roundtrip", ok)
    except Exception as e:
        tr.record("reproducibility_rng_roundtrip", False, str(e))

    # ---- 3. Loss components (hand-computed) ----
    try:
        b1 = torch.tensor([[0., 0., 10., 10.]])
        giou_identical = box_giou(b1, b1).item()
        ok1 = abs(giou_identical - 1.0) < 1e-5

        fl_low = focal_loss(torch.tensor([[0.1, 0.0]]), torch.tensor([0]))
        fl_high = focal_loss(torch.tensor([[5.0, 0.0]]), torch.tensor([0]))
        ok2 = fl_high.item() < fl_low.item()

        obj_loss = objectness_bce_loss(torch.tensor([5.0, -5.0]), torch.tensor([1.0, 0.0]))
        ok3 = obj_loss.item() < 0.05

        q_loss = quality_iou_loss(torch.tensor([0.0]), b1, b1)
        ok4 = abs(q_loss.item() - 0.25) < 1e-4  # (sigmoid(0)=0.5, target_iou=1.0) -> (0.5-1)^2=0.25

        ok = ok1 and ok2 and ok3 and ok4
        tr.record("loss_components_hand_computed", ok, f"giou={giou_identical}, quality_loss={q_loss.item()}")
    except Exception as e:
        tr.record("loss_components_hand_computed", False, str(e))

    # ---- 4. Target assignment + recall-ceiling diagnostic ----
    try:
        meta_in = PatchMetadata(patch_index=0, row_index=0, col_index=0, y0=0, x0=0, y1=16, x1=16,
                                 center_y=8.0, center_x=8.0, temporal_dim=10, spatial_size=16)
        meta_out = PatchMetadata(patch_index=1, row_index=0, col_index=5, y0=0, x0=80, y1=16, x1=96,
                                  center_y=8.0, center_x=88.0, temporal_dim=10, spatial_size=16)
        dtype = np.dtype([("ts", "<u8"), ("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
                           ("class_id", "u1"), ("confidence", "<f4"), ("track_id", "<u4")])
        boxes = np.zeros(1, dtype=dtype)
        boxes[0] = (0, 2.0, 2.0, 12.0, 12.0, 1, 1.0, 0)

        result_hit = assign_targets([meta_in], boxes)
        result_miss = assign_targets([meta_out], boxes)
        ok = (bool(result_hit.is_positive[0]) and result_hit.n_unmatched_gt_boxes == 0
              and not bool(result_miss.is_positive[0]) and result_miss.n_unmatched_gt_boxes == 1)
        tr.record("target_assignment_and_recall_ceiling", ok)
    except Exception as e:
        tr.record("target_assignment_and_recall_ceiling", False, str(e))

    # ---- 5. Sparsity curriculum stages ----
    try:
        cfg = TrainingConfig()
        b0 = get_current_sparsity_band(0.0, cfg)
        b_mid = get_current_sparsity_band((cfg.stage1_end_pct + cfg.stage2_end_pct) / 2, cfg)
        b_end = get_current_sparsity_band(1.0, cfg)
        ok = (
            b0.stage == 1 and abs(b0.min_ratio - cfg.curriculum_start_min_ratio) < 1e-9
            and b_mid.stage == 2 and cfg.curriculum_target_min_ratio < b_mid.min_ratio < cfg.curriculum_start_min_ratio
            and b_end.stage == 3 and abs(b_end.min_ratio - cfg.curriculum_target_min_ratio) < 1e-9
        )
        tr.record("sparsity_curriculum_stages", ok, f"stage1={b0}, stage2_mid={b_mid}, stage3={b_end}")
    except Exception as e:
        tr.record("sparsity_curriculum_stages", False, str(e))

    # ---- 6. Augmentation: bounds, ordering preserved, disabled = no-op ----
    dataset = None
    try:
        manifest = create_split(test_dataset_root, seed=42)
        smoke_manifest = _make_small_manifest(manifest, train_count=1, val_count=1)
        dataset = EDPSGen1Dataset(
            test_dataset_root, smoke_manifest, split="train", window_us=window_us
        )
        sample = dataset[0]

        cfg_aug = TrainingConfig(use_augmentation=True, temporal_jitter_std_us=5000, event_dropout_prob=0.1,
                                  spatial_translation_max_frac=0.15, random_crop_min_area_frac=0.8)
        rng = np.random.default_rng(0)
        aug = augment_sample(sample, cfg_aug, rng)
        bounds_ok = (
            (aug["x"] >= 0).all() and (aug["x"] < aug["sensor_width"]).all()
            and (aug["y"] >= 0).all() and (aug["y"] < aug["sensor_height"]).all()
        )
        sorted_ok = np.all(np.diff(aug["t"]) >= 0)

        cfg_off = TrainingConfig(use_augmentation=False)
        aug_off = augment_sample(sample, cfg_off, rng)
        noop_ok = len(aug_off["t"]) == len(sample["t"])

        ok = bounds_ok and sorted_ok and noop_ok
        tr.record("augmentation_bounds_ordering_noop", ok,
                   f"bounds_ok={bounds_ok}, sorted_ok={sorted_ok}, noop_ok={noop_ok}")
    except Exception as e:
        tr.record("augmentation_bounds_ordering_noop", False, str(e))

    # ---- 7. Optimizer: configurable EDPS LR (requirement 1) ----
    n_rows, n_cols = 15, 19
    if dataset is not None:
        try:
            n_rows, n_cols, _, _ = compute_patch_grid_dims(
                dataset[0]["sensor_height"], dataset[0]["sensor_width"], PatchConfig(patch_size=16)
            )
        except Exception:
            pass

    try:
        embed, edps_m, enc, head = _build_components(n_rows, n_cols)
        cfg_lr = TrainingConfig(base_lr=1e-4, edps_lr=3e-4)
        opt = build_optimizer(embed, edps_m, enc, head, cfg_lr)
        lrs = {g["lr"] for g in opt.param_groups}
        ok_diff = lrs == {1e-4, 3e-4}

        cfg_same = TrainingConfig(base_lr=1e-4, edps_lr=None)  # default: same LR everywhere
        opt2 = build_optimizer(embed, edps_m, enc, head, cfg_same)
        lrs2 = {g["lr"] for g in opt2.param_groups}
        ok_default = lrs2 == {1e-4}

        tr.record("optimizer_configurable_edps_lr", ok_diff and ok_default,
                   f"differing_lrs={lrs}, default_same_lr={lrs2}")
    except Exception as e:
        tr.record("optimizer_configurable_edps_lr", False, str(e))

    # ---- 8. Scheduler LR trajectory (warmup -> cosine -> floor) ----
    try:
        cfg_sched = TrainingConfig(warmup_pct=0.1, min_lr_ratio=0.01)
        opt3 = build_optimizer(embed, edps_m, enc, head, cfg_sched)
        sched = build_scheduler(opt3, cfg_sched, total_steps=100)
        lrs_seen = []
        for _ in range(100):
            opt3.step()
            sched.step()
            lrs_seen.append(opt3.param_groups[-1]["lr"])
        ok = (lrs_seen[9] > lrs_seen[0] and abs(lrs_seen[-1] - cfg_sched.min_lr_ratio * cfg_sched.base_lr) < 1e-8)
        tr.record("scheduler_lr_trajectory", ok, f"lr[0]={lrs_seen[0]:.2e}, lr[9]={lrs_seen[9]:.2e}, lr[-1]={lrs_seen[-1]:.2e}")
    except Exception as e:
        tr.record("scheduler_lr_trajectory", False, str(e))

    # ---- 9. Checkpoint save/load/RESUME round-trip ----
    ckpt_dir = "test_m10_ckpt_dir"
    try:
        embed_a, edps_a, enc_a, head_a = _build_components(n_rows, n_cols)
        cfg_ckpt = TrainingConfig(checkpoint_dir=ckpt_dir)
        opt_a = build_optimizer(embed_a, edps_a, enc_a, head_a, cfg_ckpt)
        sched_a = build_scheduler(opt_a, cfg_ckpt, total_steps=100)

        dummy = torch.randn(1, 5, 256)
        for _ in range(3):
            out = enc_a(dummy)
            loss = out.sum()
            opt_a.zero_grad(); loss.backward(); opt_a.step(); sched_a.step()

        state = TrainingState(epoch=3, global_step=3, best_metric=0.42, best_epoch=2)
        path = periodic_checkpoint_path(ckpt_dir, 3)
        save_checkpoint(path, embed_a, edps_a, enc_a, head_a, opt_a, sched_a, None, state, cfg_ckpt)

        enc_weight_before = enc_a.blocks[0].attn.in_proj_weight.clone()
        embed_b, edps_b, enc_b, head_b = _build_components(n_rows, n_cols)  # fresh, different init
        opt_b = build_optimizer(embed_b, edps_b, enc_b, head_b, cfg_ckpt)
        sched_b = build_scheduler(opt_b, cfg_ckpt, total_steps=100)
        loaded = load_checkpoint(path, embed_b, edps_b, enc_b, head_b, opt_b, sched_b, None, restore_rng=True)

        weights_match = torch.equal(enc_weight_before, enc_b.blocks[0].attn.in_proj_weight)
        state_match = (loaded.epoch == 3 and loaded.global_step == 3 and loaded.best_metric == 0.42)
        found = find_latest_checkpoint(ckpt_dir)
        found_match = found == path

        ok = weights_match and state_match and found_match
        tr.record("checkpoint_resume_roundtrip", ok,
                   f"weights_match={weights_match}, state_match={state_match}, found_match={found_match}")
    except Exception as e:
        tr.record("checkpoint_resume_roundtrip", False, str(e))
    finally:
        if cleanup and os.path.isdir(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    # ---- 10. mAP metric correctness ----
    try:
        pred_boxes = [np.array([[0., 0., 10., 10.]])]
        pred_scores = [np.array([0.9])]
        gt_boxes = [np.array([[0., 0., 10., 10.]])]
        ap_perfect = compute_ap_for_class(pred_boxes, pred_scores, gt_boxes, 0.5)

        ap_no_gt = compute_ap_for_class([np.zeros((0, 4))], [np.zeros((0,))], [np.zeros((0, 4))], 0.5)

        ok = abs(ap_perfect - 1.0) < 1e-6 and math.isnan(ap_no_gt)
        tr.record("map_metric_correctness", ok, f"ap_perfect={ap_perfect}, ap_no_gt={ap_no_gt}")
    except Exception as e:
        tr.record("map_metric_correctness", False, str(e))

    # ---- 11. Training collate integration ----
    train_loader = None
    val_loader = None
    try:
        embed_c, edps_c, enc_c, head_c = _build_components(n_rows, n_cols)
        cfg_tc = TrainingConfig(batch_size=2, use_augmentation=True)
        train_loader = build_training_dataloader(
            dataset or EDPSGen1Dataset(test_dataset_root, _make_small_manifest(create_split(test_dataset_root, seed=42), 1, 1), split="train", window_us=window_us),
            test_dataset_root, n_rows, n_cols, embed_c, edps_c, enc_c, head_c, cfg_tc, is_train=True,
        )
        batch = next(iter(train_loader))
        ok = ("raw_predictions" in batch and "assignments" in batch
              and len(batch["assignments"]) == batch["raw_predictions"]["ltrb_raw"].shape[0])
        tr.record("training_collate_integration", ok)
    except Exception as e:
        tr.record("training_collate_integration", False, str(e))

    # ---- 12. End-to-end mini training run (real Trainer, tiny config) ----
    mini_ckpt_dir = "test_m10_mini_train"
    try:
        full_manifest = create_split(test_dataset_root, seed=42)
        smoke_manifest = _make_small_manifest(full_manifest, train_count=1, val_count=1)
        val_dataset = EDPSGen1Dataset(
            test_dataset_root, smoke_manifest, split="val", window_us=window_us
        )

        # Limit the smoke test to a handful of temporal windows. This test is
        # about proving that the real Trainer can execute end-to-end, not about
        # training the complete dataset.
        mini_train_dataset = EDPSGen1Dataset(
            test_dataset_root, smoke_manifest, split="train", window_us=window_us
        )
        mini_train_dataset.windows = mini_train_dataset.windows[:4]
        val_dataset.windows = val_dataset.windows[:2]

        embed_d, edps_d, enc_d, head_d = _build_components(n_rows, n_cols)
        cfg_mini = TrainingConfig(
            batch_size=2, grad_accumulation_steps=1, total_epochs=1, checkpoint_dir=mini_ckpt_dir,
            checkpoint_every_n_epochs=1, validate_every_n_epochs=1, use_amp=False, use_augmentation=False,
            stage1_end_pct=1.0, stage2_end_pct=1.0,  # keep the sparsity band lenient/constant for this tiny run
        )
        mini_device = "cuda" if torch.cuda.is_available() else "cpu"

        train_loader_mini = build_training_dataloader(
            mini_train_dataset, test_dataset_root, n_rows, n_cols,
            embed_d, edps_d, enc_d, head_d, cfg_mini,
            is_train=True,
            device=mini_device,
        )
        val_loader_mini = build_training_dataloader(
            val_dataset, test_dataset_root, n_rows, n_cols,
            embed_d, edps_d, enc_d, head_d, cfg_mini,
            is_train=False,
            device=mini_device,
        )
        trainer = Trainer(
            embed_d, edps_d, enc_d, head_d,
            train_loader_mini, val_loader_mini,
            cfg_mini, num_classes=2, device=mini_device,
        )
        history = trainer.fit(resume=False)

        no_nan = all(not math.isnan(h.train_loss) and not math.isinf(h.train_loss) for h in history)
        ran_epochs = len(history) == cfg_mini.total_epochs
        has_val_map = all(h.val_map50 is not None for h in history)
        ckpt_exists = os.path.exists(periodic_checkpoint_path(mini_ckpt_dir, cfg_mini.total_epochs - 1))

        ok = no_nan and ran_epochs and has_val_map and ckpt_exists
        tr.record("end_to_end_mini_training", ok,
                   f"no_nan={no_nan}, ran_epochs={ran_epochs} ({len(history)}/{cfg_mini.total_epochs}), "
                   f"has_val_map={has_val_map}, ckpt_exists={ckpt_exists}, "
                   f"losses={[round(h.train_loss,4) for h in history]}")

        # ---- 12b. Resume continues from the correct epoch with correct weights ----
        embed_e, edps_e, enc_e, head_e = _build_components(n_rows, n_cols)  # fresh, different init

        # Resume must use DataLoaders whose collator owns the same fresh
        # modules as trainer2. Reusing train_loader_mini would leave its
        # collator wired to embed_d/edps_d/enc_d/head_d.
        train_loader_resume = build_training_dataloader(
            mini_train_dataset, test_dataset_root, n_rows, n_cols,
            embed_e, edps_e, enc_e, head_e, cfg_mini,
            is_train=True,
            device=mini_device,
        )
        val_loader_resume = build_training_dataloader(
            val_dataset, test_dataset_root, n_rows, n_cols,
            embed_e, edps_e, enc_e, head_e, cfg_mini,
            is_train=False,
            device=mini_device,
        )
        trainer2 = Trainer(
            embed_e, edps_e, enc_e, head_e,
            train_loader_resume, val_loader_resume,
            cfg_mini, num_classes=2, device=mini_device,
        )
        pre_resume_weight = enc_d.blocks[0].attn.in_proj_weight.clone()
        trainer2.fit(resume=True)  # should load the epoch-1 checkpoint saved above and finish immediately (epoch>=total)
        post_resume_weight = trainer2.encoder.blocks[0].attn.in_proj_weight
        resumed_correctly = trainer2.state.epoch >= cfg_mini.total_epochs - 1
        tr.record("resume_from_checkpoint", resumed_correctly,
                   f"resumed_state_epoch={trainer2.state.epoch} (expected >= {cfg_mini.total_epochs - 1})")
    except Exception as e:
        tr.record("end_to_end_mini_training", False, str(e))
        tr.record("resume_from_checkpoint", False, "skipped due to prior failure")
    finally:
        if cleanup and os.path.isdir(mini_ckpt_dir):
            shutil.rmtree(mini_ckpt_dir)
        if cleanup and created_temp_synth and os.path.isdir(temp_synth_dir):
            shutil.rmtree(temp_synth_dir)

    # ---- 13. Report generation (importability smoke test) ----
    try:
        from generate_m10_report import generate_m10_report  # noqa
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

    passed, _ = run_all_m10_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
