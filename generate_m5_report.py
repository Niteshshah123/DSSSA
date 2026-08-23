"""
================================================================================
M5 Research Log Report Generator
================================================================================
Runs the M5 unit test suite, computes patch/activity statistics aggregated
over sampled windows, the computational complexity analysis, saves an
example patch visualization, and writes M5_report.json / M5_report.md.

Usage:
    python generate_m5_report.py --dataset_root /path/to/dataset --output_dir /content/m5_report
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
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_config import PatchConfig
from patch_generator import PatchGenerator, estimate_patch_generation_cost
from patch_visualization import visualize_patches
from test_m5_patches import run_all_m5_tests


def generate_m5_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    patch_size: int = 16,
    padding_mode: str = "zero",
    adjacency_mode: str = "4",
    num_sample_windows: int = 30,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_m5_tests(dataset_root, window_us=window_us)

    voxel_config = VoxelGridConfig()
    patch_config = PatchConfig(patch_size=patch_size, padding_mode=padding_mode, adjacency_mode=adjacency_mode)

    manifest = create_split(dataset_root, seed=seed)
    dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)

    preprocessor = EventPreprocessor(PreprocessingConfig())
    voxel_gen = VoxelGridGenerator(voxel_config)
    patch_gen = PatchGenerator(patch_config, voxel_config)

    n_sample = min(num_sample_windows, len(dataset))
    n_active_list = []
    avg_events_per_patch_list = []
    processing_times = []
    example_result = None
    example_voxel = None

    ACTIVE_THRESHOLD = 0.0  # a patch counts as "active" if it has ANY recorded activity

    for i in range(n_sample):
        sample = dataset[i]
        t0 = time.perf_counter()
        cleaned = preprocessor.process(sample, dataset_root=dataset_root)
        voxel = voxel_gen.generate({
            "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
            "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
            "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
        })
        result = patch_gen.generate(voxel)
        elapsed = time.perf_counter() - t0
        processing_times.append(elapsed)

        totals = [s.total_event_count for s in result.stats]
        n_active_list.append(sum(1 for v in totals if v > ACTIVE_THRESHOLD))
        avg_events_per_patch_list.append(float(np.mean(totals)) if totals else 0.0)

        if example_result is None:
            example_result = result
            example_voxel = voxel

    n_patches = len(example_result.metadata) if example_result else 0
    cost = estimate_patch_generation_cost(
        n_patches, patch_config.patch_size, voxel_config.num_channels,
        example_voxel.shape[1], example_voxel.shape[2],
    ) if example_result else {}

    viz_path = os.path.join(output_dir, "patch_visualization_example.png")
    if example_result is not None:
        visualize_patches(
            example_voxel, example_result.metadata, example_result.stats, voxel_config, patch_config,
            example_result.n_rows, example_result.n_cols, out_path=viz_path,
        )

    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M5 - Patch Generation",
        "config": {
            "patch_size": patch_config.patch_size,
            "padding_mode": patch_config.padding_mode,
            "adjacency_mode": patch_config.adjacency_mode,
            "voxel_num_bins": voxel_config.num_bins,
            "voxel_polarity_encoding": voxel_config.polarity_encoding,
        },
        "patch_grid": {
            "n_rows": example_result.n_rows if example_result else None,
            "n_cols": example_result.n_cols if example_result else None,
            "n_patches": n_patches,
        },
        "statistics": {
            "num_windows_sampled": n_sample,
            "avg_active_patches_per_window": float(np.mean(n_active_list)) if n_active_list else 0.0,
            "std_active_patches_per_window": float(np.std(n_active_list)) if n_active_list else 0.0,
            "pct_active_patches": (100.0 * float(np.mean(n_active_list)) / n_patches) if n_patches else 0.0,
            "avg_events_per_patch": float(np.mean(avg_events_per_patch_list)) if avg_events_per_patch_list else 0.0,
            "avg_processing_time_ms": float(np.mean(processing_times) * 1000) if processing_times else 0.0,
            "std_processing_time_ms": float(np.std(processing_times) * 1000) if processing_times else 0.0,
        },
        "computational_complexity": cost,
        "example_visualization": viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
    }

    json_path = os.path.join(output_dir, "M5_report.json")
    md_path = os.path.join(output_dir, "M5_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M5 - Patch Generation Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Patch Grid\n\n")
        for k, v in report["patch_grid"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Statistics (aggregated over {n_sample} sampled windows)\n\n")
        for k, v in report["statistics"].items():
            f.write(f"- **{k}**: {v}\n")
        f.write(
            "\n_Note: 'active' here means non-zero recorded activity -- this is a DESCRIPTIVE "
            "count only. M5 performs no pruning; EDPS (M7) is solely responsible for deciding "
            "which patches to retain._\n"
        )

        f.write("\n## Computational Complexity Analysis\n\n")
        for k, v in cost.items():
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
    ap.add_argument("--output_dir", default="./m5_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--patch_size", type=int, default=16, choices=[8, 16, 32])
    ap.add_argument("--padding_mode", default="zero", choices=["zero", "ignore"])
    ap.add_argument("--adjacency_mode", default="4", choices=["4", "8"])
    ap.add_argument("--num_sample_windows", type=int, default=30)
    args = ap.parse_args()

    generate_m5_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        patch_size=args.patch_size, padding_mode=args.padding_mode,
        adjacency_mode=args.adjacency_mode, num_sample_windows=args.num_sample_windows,
    )
