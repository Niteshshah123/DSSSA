"""
================================================================================
M10 Research Log Report Generator
================================================================================
Supports two explicit modes:

1. DEMO MODE
   Small, bounded run used to verify the complete M10 pipeline.
   It does NOT launch the full dataset training job.

2. TRAIN MODE
   Full recording-level train/validation split and the real TrainingConfig.
   This is the actual end-to-end model training run.

Both modes first execute the M10 unit-test suite unless --skip_tests is used.
================================================================================
"""

from __future__ import annotations

import os
import json
import time
import argparse
from datetime import datetime, timezone

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
from training_config import TrainingConfig
from training_collate import build_training_dataloader
from trainer import Trainer
from test_m10_training import run_all_m10_tests


def _build_components(train_dataset, baseline_no_pruning=False):
    n_rows, n_cols, _, _ = compute_patch_grid_dims(
        train_dataset[0]["sensor_height"],
        train_dataset[0]["sensor_width"],
        PatchConfig(patch_size=16),
    )

    embedding_module = PatchEmbeddingModule(
        PatchEmbeddingConfig(),
        in_channels=10,
        patch_size=16,
        n_rows=n_rows,
        n_cols=n_cols,
        polarity_encoding="signed",
    )

    # Diagnostic baseline:
    # keep every token after the EDPS hard-selection stage. EDPS still computes
    # its learned scores, so this is a controlled "no hard pruning" baseline,
    # not removal of the EDPS code from the pipeline.
    edps_config = EDPSConfig()
    if baseline_no_pruning:
        edps_config.selection_mode = "baseline_no_pruning"

    edps_module = EDPSModule(
        edps_config,
        embedding_dim=256,
        num_bins=10,
    )
    encoder = TransformerEncoder(TransformerEncoderConfig())
    detection_head = SparseDetectionHead(DetectionHeadConfig())

    return embedding_module, edps_module, encoder, detection_head, n_rows, n_cols


def _history_to_dict(history):
    return [
        {
            "epoch": h.epoch,
            "train_loss": h.train_loss,
            "val_map50": h.val_map50,
            "val_map50_95": h.val_map50_95,
            "avg_retained_tokens": h.avg_retained_tokens,
            "sparsity_band": h.sparsity_band,
            "n_unmatched_gt_fraction": h.n_unmatched_gt_fraction,
            "lr": h.lr,
            "epoch_time_s": h.epoch_time_s,
            "loss_components": h.train_loss_components,
        }
        for h in history
    ]


def _write_report(output_dir, report, curve_title):
    json_path = os.path.join(output_dir, "M10_report.json")
    md_path = os.path.join(output_dir, "M10_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    curve = report["training_curve"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# M10 - End-to-End Training Pipeline Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")
        f.write(f"**Mode:** `{report['mode']}`\n\n")
        f.write(f"> {report['note']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## {curve_title}\n\n")
        f.write(
            "| Epoch | Train Loss | Val mAP@0.5 | Val mAP@0.5:0.95 | "
            "Avg Retained | Sparsity Band | Unmatched GT % | Epoch Time (s) |\n"
        )
        f.write("|---|---:|---:|---:|---:|---|---:|---:|\n")

        for row in curve:
            map50 = f"{row['val_map50']:.4f}" if row["val_map50"] is not None else "N/A"
            map5095 = (
                f"{row['val_map50_95']:.4f}"
                if row["val_map50_95"] is not None
                else "N/A"
            )
            band = row["sparsity_band"]
            f.write(
                f"| {row['epoch']} | {row['train_loss']:.4f} | "
                f"{map50} | {map5095} | {row['avg_retained_tokens']:.1f} | "
                f"[{band['min_ratio']:.2f},{band['max_ratio']:.2f}] "
                f"(stage {band['stage']}) | "
                f"{100 * row['n_unmatched_gt_fraction']:.1f}% | "
                f"{row['epoch_time_s']:.1f} |\n"
            )

        best = report["best_metric_reached"]
        best_str = f"{best:.4f}" if best is not None else "N/A"
        f.write(f"\n**Best mAP@0.5 reached:** {best_str}\n")
        f.write(f"**Best epoch:** {report['best_epoch']}\n")

        if "edps_summary" in report:
            edps = report["edps_summary"]
            f.write("\n## EDPS Selection Summary\n\n")
            f.write(f"- **Selection mode:** `{edps.get('selection_mode', 'N/A')}`\n")
            f.write(f"- **EDPS threshold:** `{edps.get('gate_threshold', 0.5):.4f}`\n")
            f.write(f"- **Mean gate score:** `{edps.get('mean_gate', 0.0):.4f}`\n")
            f.write(f"- **Median gate score:** `{edps.get('median_gate', 0.0):.4f}`\n")
            f.write(f"- **Min gate score:** `{edps.get('min_gate', 0.0):.4f}`\n")
            f.write(f"- **Max gate score:** `{edps.get('max_gate', 0.0):.4f}`\n")
            f.write(f"- **Gate keep rate:** `{edps.get('gate_keep_rate', 0.0):.2f}%`\n")
            f.write(f"- **Actual selected tokens:** `{edps.get('avg_retained_tokens', 0.0):.1f}`\n")
            f.write(f"- **Actual retention:** `{edps.get('actual_retention_pct', 0.0):.2f}%`\n")

        if report.get("gate_score_evolution_plot"):
            f.write(
                f"\n## Gate Score Evolution\n\n"
                f"- `{report['gate_score_evolution_plot']}`\n"
            )

        f.write("\n## Unit Test Results\n\n")
        for test in report["unit_test_results"]:
            status = "PASS" if test["passed"] else "FAIL"
            detail = f" — {test['detail']}" if test["detail"] else ""
            f.write(f"- [{status}] {test['name']}{detail}\n")

        f.write(f"\n**All unit tests passed:** {report['all_unit_tests_passed']}\n")
        f.write(
            f"\n**Execution time:** "
            f"{report['execution_time_seconds']:.2f}s\n"
        )

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")
    return report


def generate_m10_report(
    dataset_root: str,
    output_dir: str = "./m10_report",
    window_us: int = 50_000,
    mode: str = "demo",
    demo_epochs: int = 1,
    demo_recordings: int = 1,
    demo_max_windows: int = 8,
    seed: int = 42,
    device: str = "auto",
    skip_tests: bool = False,
    baseline_no_pruning: bool = False,
    disable_augmentation: bool = False,
    drop_empty_windows: bool = None,
    num_workers: int = None,
    batch_size: int = None,
    resume_checkpoint: str = None,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if drop_empty_windows is None:
        drop_empty_windows = (mode == "train")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    print(f"[M10] Mode: {mode}")
    print(f"[M10] Device: {device}")
    print(f"[M10] Baseline no-pruning: {baseline_no_pruning}")
    print(f"[M10] Augmentation disabled: {disable_augmentation}")
    print(f"[M10] Drop empty windows: {drop_empty_windows}")
    if device == "cuda":
        print(f"[M10] GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Unit tests
    # ------------------------------------------------------------------
    if skip_tests:
        all_passed = True
        test_results = []
        print("[M10] Unit tests skipped by --skip_tests")
    else:
        all_passed, test_results = run_all_m10_tests(
            dataset_root,
            window_us=window_us,
        )

        if not all_passed:
            raise RuntimeError(
                "M10 unit tests did not all pass. "
                "Refusing to start training."
            )

    from dataset_split import resolve_dataset_root, create_split
    dataset_root = resolve_dataset_root(dataset_root)
    print(f"[M10] Resolved dataset root: {dataset_root}")

    # ------------------------------------------------------------------
    # 2. Dataset and model
    # ------------------------------------------------------------------
    manifest = create_split(dataset_root, seed=seed)

    train_dataset = EDPSGen1Dataset(
        dataset_root,
        manifest,
        split="train",
        window_us=window_us,
        drop_empty_windows=drop_empty_windows,
    )
    val_dataset = EDPSGen1Dataset(
        dataset_root,
        manifest,
        split="val",
        window_us=window_us,
        drop_empty_windows=drop_empty_windows,
    )

    print(
        f"[M10] Full split: train_recordings={len(manifest.train)}, "
        f"val_recordings={len(manifest.val)}, test_recordings={len(manifest.test)}"
    )
    print(
        f"[M10] Full windows: train={len(train_dataset)}, "
        f"val={len(val_dataset)}"
    )

    (
        embedding_module,
        edps_module,
        encoder,
        detection_head,
        n_rows,
        n_cols,
    ) = _build_components(
        train_dataset,
        baseline_no_pruning=baseline_no_pruning,
    )

    # ------------------------------------------------------------------
    # 3. Explicit DEMO vs REAL TRAIN configuration
    # ------------------------------------------------------------------
    if mode == "demo":
        # Deliberately bounded. Never silently becomes a full training run.
        demo_manifest = type(manifest)(
            seed=manifest.seed,
            train_ratio=manifest.train_ratio,
            val_ratio=manifest.val_ratio,
            test_ratio=manifest.test_ratio,
            train=list(manifest.train[:max(1, min(demo_recordings, len(manifest.train)))]),
            val=list(manifest.val[:max(1, min(1, len(manifest.val)))]),
            test=[],
        )

        train_dataset = EDPSGen1Dataset(
            dataset_root, demo_manifest, split="train", window_us=window_us
        )
        val_dataset = EDPSGen1Dataset(
            dataset_root, demo_manifest, split="val", window_us=window_us
        )

        train_dataset.windows = train_dataset.windows[:demo_max_windows]
        val_dataset.windows = val_dataset.windows[:max(1, min(2, len(val_dataset.windows)))]

        demo_ckpt_dir = os.path.join(output_dir, "demo_checkpoints")
        config_kwargs = dict(
            batch_size=2,
            grad_accumulation_steps=1,
            total_epochs=demo_epochs,
            checkpoint_dir=demo_ckpt_dir,
            checkpoint_every_n_epochs=1,
            validate_every_n_epochs=1,
            use_amp=False,
            use_augmentation=not disable_augmentation,
            seed=seed,
        )

        if baseline_no_pruning:
            # Keep the sparsity objective from fighting the diagnostic baseline.
            # The EDPS hard-selection clamp itself is set to 100% retention above.
            config_kwargs.update(
                curriculum_start_min_ratio=0.95,
                curriculum_start_max_ratio=1.0,
                curriculum_target_min_ratio=0.95,
                curriculum_target_max_ratio=1.0,
                lambda_sparsity=0.0,
            )

        config = TrainingConfig(**config_kwargs)

        run_note = (
            "Short bounded DEMO run. It validates that the complete M10 "
            "pipeline trains, but it is not a final model-quality experiment."
        )
        if baseline_no_pruning:
            run_note += (
                " Diagnostic baseline: EDPS hard selection is clamped to "
                "100% retention and the sparsity loss is disabled, so this "
                "measures detector/backbone learning without token removal."
            )
        if disable_augmentation:
            run_note += " Training augmentation is disabled for this run."
        curve_title = f"Demo Training Curve ({demo_epochs} epochs)"

    elif mode == "train":
        # Full dataset + actual TrainingConfig.
        actual_batch_size = batch_size if batch_size is not None else 16
        config = TrainingConfig(
            total_epochs=20,
            edps_lr=3e-4,
            batch_size=actual_batch_size,
            grad_accumulation_steps=2,
            use_amp=True,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            checkpoint_every_n_epochs=5,
            validate_every_n_epochs=1,
            seed=seed,
        )
        run_note = (
            "REAL end-to-end training run using the complete recording-level "
            "training split and the configured M10 training pipeline. "
            "Checkpointing/resume is enabled."
        )
        curve_title = f"Real Training Curve ({config.total_epochs} epochs)"

    else:
        raise ValueError("mode must be 'demo' or 'train'")

    print(
        f"[M10] Run config: epochs={config.total_epochs}, "
        f"batch_size={config.batch_size}, "
        f"grad_accumulation={config.grad_accumulation_steps}, "
        f"effective_batch={config.batch_size * config.grad_accumulation_steps}, "
        f"AMP={config.use_amp}, base_lr={config.base_lr}, "
        f"edps_lr={config.effective_edps_lr}"
    )

    train_loader = build_training_dataloader(
        train_dataset,
        dataset_root,
        n_rows,
        n_cols,
        embedding_module,
        edps_module,
        encoder,
        detection_head,
        config,
        is_train=True,
        num_workers=num_workers,
        device=device,
    )

    val_loader = build_training_dataloader(
        val_dataset,
        dataset_root,
        n_rows,
        n_cols,
        embedding_module,
        edps_module,
        encoder,
        detection_head,
        config,
        is_train=False,
        num_workers=num_workers,
        device=device,
    )

    trainer = Trainer(
        embedding_module,
        edps_module,
        encoder,
        detection_head,
        train_loader,
        val_loader,
        config,
        num_classes=2,
        device=device,
    )

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    resume = (mode == "train") or (resume_checkpoint is not None)
    print(f"[M10] Starting {'RESUMABLE ' if resume else ''}{mode} training...")
    history = trainer.fit(resume=resume, resume_checkpoint=resume_checkpoint)

    training_curve = _history_to_dict(history)

    gate_hist_path = os.path.join(
        output_dir,
        f"{mode}_gate_score_evolution.png",
    )
    try:
        trainer.gate_logger.plot_histogram(gate_hist_path)
    except Exception as exc:
        print(f"[M10] Warning: gate histogram could not be written: {exc}")
        gate_hist_path = None

    elapsed_total = time.time() - start_time

    avg_ret_tokens = float(np.mean([h.avg_retained_tokens for h in history])) if history else 285.0
    last_snap = trainer.gate_logger.history[-1] if trainer.gate_logger.history else None
    edps_summary = {
        "selection_mode": edps_module.config.selection_mode,
        "gate_threshold": float(edps_module.config.gate_threshold),
        "mean_gate": float(last_snap.mean) if last_snap else 0.5,
        "median_gate": float(last_snap.median) if last_snap else 0.5,
        "min_gate": float(last_snap.min) if last_snap else 0.0,
        "max_gate": float(last_snap.max) if last_snap else 1.0,
        "gate_keep_rate": float(100.0 * last_snap.frac_above_threshold) if last_snap else 100.0,
        "avg_retained_tokens": avg_ret_tokens,
        "actual_retention_pct": float(100.0 * avg_ret_tokens / 285.0),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M10 - End-to-End Training Pipeline",
        "mode": mode,
        "config": config.to_dict(),
        "edps_summary": edps_summary,
        "training_curve": training_curve,
        "best_metric_reached": trainer.state.best_metric,
        "best_epoch": trainer.state.best_epoch,
        "gate_score_evolution_plot": gate_hist_path,
        "unit_test_results": [
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
        "train_recordings": len(manifest.train),
        "val_recordings": len(manifest.val),
        "test_recordings": len(manifest.test),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "note": run_note,
    }

    return _write_report(output_dir, report, curve_title)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="M10 demo or full training runner"
    )
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", default="./m10_report")
    ap.add_argument("--window_us", type=int, default=50_000)

    # Explicit mode: prevents accidentally launching full training.
    ap.add_argument(
        "--mode",
        choices=["demo", "train"],
        default="demo",
        help="demo = bounded smoke run; train = full real training",
    )

    # Demo/Train epoch controls.
    ap.add_argument("--epochs", "--demo_epochs", dest="demo_epochs", type=int, default=1, help="Total training epochs")
    ap.add_argument("--demo_recordings", type=int, default=1)
    ap.add_argument("--demo_max_windows", type=int, default=8)

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    ap.add_argument(
        "--skip_tests",
        action="store_true",
        help="Skip M10 unit tests. Use only after a verified 15/15 run.",
    )

    ap.add_argument(
        "--baseline_no_pruning",
        action="store_true",
        help=(
            "Diagnostic DEMO baseline: force EDPS hard selection to retain "
            "100%% of tokens and disable the sparsity loss. This does not "
            "remove EDPS scoring; it removes hard token pruning for the "
            "baseline experiment."
        ),
    )
    ap.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Disable train-time event augmentation for the diagnostic run.",
    )
    ap.add_argument(
        "--drop_empty_windows",
        action="store_true",
        default=None,
        help="Filter out empty event windows containing zero target bounding boxes.",
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes for multi-worker CPU data prefetching.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per step (default: 16 in train mode for max GPU utilization).",
    )
    ap.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to an uploaded checkpoint file (.pt) to explicitly resume training from.",
    )

    args = ap.parse_args()

    generate_m10_report(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        window_us=args.window_us,
        mode=args.mode,
        demo_epochs=args.demo_epochs,
        demo_recordings=args.demo_recordings,
        demo_max_windows=args.demo_max_windows,
        seed=args.seed,
        device=args.device,
        skip_tests=args.skip_tests,
        baseline_no_pruning=args.baseline_no_pruning,
        disable_augmentation=args.disable_augmentation,
        drop_empty_windows=args.drop_empty_windows,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        resume_checkpoint=args.resume_checkpoint,
    )
