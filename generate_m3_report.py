"""
================================================================================
M3 Research Log Report Generator
================================================================================
Runs the M3 unit test suite, computes voxelization cost statistics over a
sample of real windows (from the M2 split), saves an example voxel grid
visualization, and writes a report (JSON + Markdown) summarizing:
  - voxel grid configuration
  - per-module cost ledger entry (time/space complexity, estimated FLOPs)
  - measured statistics over sampled windows
  - unit test results
  - execution time

Usage:
    python generate_m3_report.py --dataset_root /path/to/dataset --output_dir /content/m3_report
================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np

from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from voxel_grid import VoxelGridConfig, VoxelGridGenerator, estimate_voxelization_cost
from voxel_visualization import visualize_voxel_grid
from test_voxel_grid import run_all_voxel_tests


def generate_m3_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    num_bins: int = 10,
    temporal_mode: str = "bilinear",
    polarity_encoding: str = "signed",
    num_sample_windows: int = 50,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_voxel_tests(dataset_root, window_us=window_us)

    config = VoxelGridConfig(num_bins=num_bins, temporal_mode=temporal_mode, polarity_encoding=polarity_encoding)
    generator = VoxelGridGenerator(config)

    manifest = create_split(dataset_root, seed=seed)
    dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)

    n_sample = min(num_sample_windows, len(dataset))
    voxelization_times = []
    event_counts = []
    flops_estimates = []
    example_sample = None
    for i in range(n_sample):
        sample = dataset[i]
        if example_sample is None and len(sample["boxes"]) > 0:
            example_sample = sample

        t0 = time.perf_counter()
        _ = generator.generate(sample)
        elapsed = time.perf_counter() - t0

        n_events = len(sample["t"])
        voxelization_times.append(elapsed)
        event_counts.append(n_events)
        flops_estimates.append(
            estimate_voxelization_cost(n_events, config, sample["sensor_height"], sample["sensor_width"])["estimated_flops"]
        )

    if example_sample is None:
        example_sample = dataset[0]

    example_voxel = generator.generate(example_sample)
    viz_path = os.path.join(output_dir, "example_voxel_grid.png")
    visualize_voxel_grid(example_voxel, config, out_path=viz_path, boxes=example_sample["boxes"])

    cost_ledger_entry = estimate_voxelization_cost(
        int(np.mean(event_counts)) if event_counts else 0, config,
        example_sample["sensor_height"], example_sample["sensor_width"],
    )

    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M3 - Voxel Grid Generation",
        "config": {
            "num_bins": num_bins,
            "temporal_mode": temporal_mode,
            "polarity_encoding": polarity_encoding,
            "output_channels": config.num_channels,
            "window_us": window_us,
        },
        "cost_ledger_entry": cost_ledger_entry,
        "measured_statistics": {
            "num_windows_sampled": n_sample,
            "avg_events_per_window": float(np.mean(event_counts)) if event_counts else 0.0,
            "avg_voxelization_time_ms": float(np.mean(voxelization_times) * 1000) if voxelization_times else 0.0,
            "std_voxelization_time_ms": float(np.std(voxelization_times) * 1000) if voxelization_times else 0.0,
            "avg_estimated_flops": float(np.mean(flops_estimates)) if flops_estimates else 0.0,
        },
        "example_visualization": viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
    }

    json_path = os.path.join(output_dir, "M3_report.json")
    md_path = os.path.join(output_dir, "M3_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# M3 - Voxel Grid Generation Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Cost Ledger Entry (for cross-module FLOPs/latency comparison)\n\n")
        for k, v in cost_ledger_entry.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Measured Statistics\n\n")
        for k, v in report["measured_statistics"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Example Visualization\n\n- `{viz_path}`\n")

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
    ap.add_argument("--output_dir", default="./m3_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--num_bins", type=int, default=10)
    ap.add_argument("--temporal_mode", default="bilinear", choices=["bilinear", "count"])
    ap.add_argument("--polarity_encoding", default="signed", choices=["signed", "separate_channels"])
    ap.add_argument("--num_sample_windows", type=int, default=50)
    args = ap.parse_args()

    generate_m3_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        num_bins=args.num_bins, temporal_mode=args.temporal_mode,
        polarity_encoding=args.polarity_encoding, num_sample_windows=args.num_sample_windows,
    )
