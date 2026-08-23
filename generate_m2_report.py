"""
================================================================================
M2 Research Log Report Generator
================================================================================
Builds the recording-level split, constructs the train/val/test datasets,
runs the M2 unit test suite, computes per-split statistics, generates a
sample batch visualization, and writes a report (JSON + Markdown) summarizing:
  - pipeline configuration (window/stride/split ratios/seed)
  - split manifest summary
  - per-split statistics (train/val/test)
  - unit test results
  - execution time

Usage:
    python generate_m2_report.py \
        --dataset_root /path/to/dataset \
        --output_dir /content/m2_report \
        --window_us 50000
================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Optional

from dataset_split import create_split, save_manifest
from gen1_dataset import EDPSGen1Dataset
from dataloader_utils import build_dataloader
from m2_dataset_statistics import compute_split_statistics
from batch_visualization import visualize_batch
from test_m2_pipeline import run_all_m2_tests


def generate_m2_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    stride_us: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    batch_size: int = 4,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # --- Unit tests first: if the pipeline is broken, fail loudly before
    #     spending time building the "real" split/datasets below. ---
    manifest_test_path = os.path.join(output_dir, "_unit_test_manifest.json")
    all_passed, test_results = run_all_m2_tests(
        dataset_root, window_us=window_us, manifest_path=manifest_test_path
    )

    # --- Build the real, persisted split ---
    manifest = create_split(
        dataset_root, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed,
    )
    manifest_path = os.path.join(output_dir, "split_manifest.json")
    save_manifest(manifest, manifest_path)

    # --- Build datasets + per-split statistics ---
    split_stats = {}
    batch_viz_path = None
    for split in ("train", "val", "test"):
        dataset = EDPSGen1Dataset(
            dataset_root, manifest, split=split, window_us=window_us, stride_us=stride_us,
        )
        split_stats[split] = compute_split_statistics(dataset)

        if split == "train":
            loader = build_dataloader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
            batch = next(iter(loader))
            batch_viz_path = os.path.join(output_dir, "sample_batch_visualization.png")
            visualize_batch(batch, out_path=batch_viz_path)

    elapsed = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M2 - Data Loading Pipeline",
        "config": {
            "window_us": window_us,
            "stride_us": stride_us or window_us,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "batch_size": batch_size,
        },
        "split_manifest_path": manifest_path,
        "split_sizes_recordings": {
            "train": len(manifest.train), "val": len(manifest.val), "test": len(manifest.test),
        },
        "per_split_statistics": split_stats,
        "sample_batch_visualization": batch_viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed, 4),
    }

    json_path = os.path.join(output_dir, "M2_report.json")
    md_path = os.path.join(output_dir, "M2_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# M2 - Data Loading Pipeline Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Split Manifest\n\n")
        f.write(f"- Manifest saved to: `{manifest_path}`\n")
        for split, n in report["split_sizes_recordings"].items():
            f.write(f"- **{split}**: {n} recordings\n")

        for split in ("train", "val", "test"):
            s = split_stats[split]
            f.write(f"\n## {split.upper()} Split Statistics\n\n")
            f.write(f"- Recordings: {s['num_recordings']}\n")
            f.write(f"- Windows: {s['num_windows']}\n")
            f.write(f"- Avg events/window: {s['avg_events_per_window']:.2f} "
                    f"(std={s['std_events_per_window']:.2f})\n")
            f.write(f"- Avg boxes/window: {s['avg_boxes_per_window']:.2f} "
                    f"(std={s['std_boxes_per_window']:.2f})\n")
            f.write(f"- Empty windows (no boxes): {s['empty_windows']} "
                    f"({100*s['empty_window_fraction']:.1f}%)\n")
            f.write(f"- Class distribution: {s['class_distribution']}\n")

        if batch_viz_path:
            f.write(f"\n## Sample Batch Visualization\n\n")
            f.write(f"- `{batch_viz_path}`\n")

        f.write("\n## Unit Test Results\n\n")
        for t in report["unit_test_results"]:
            status = "PASS" if t["passed"] else "FAIL"
            detail = f" — {t['detail']}" if t["detail"] else ""
            f.write(f"- [{status}] {t['name']}{detail}\n")

        f.write(f"\n**All tests passed: {report['all_unit_tests_passed']}**\n")
        f.write(f"\nExecution time: {report['execution_time_seconds']:.2f}s\n")

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", default="./m2_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--stride_us", type=int, default=None)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    generate_m2_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        stride_us=args.stride_us, train_ratio=args.train_ratio,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        seed=args.seed, batch_size=args.batch_size,
    )
