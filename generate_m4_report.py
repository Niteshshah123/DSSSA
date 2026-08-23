"""
================================================================================
M4 Research Log Report Generator
================================================================================
Runs the M4 unit test suite, computes the preprocessing summary ledger
(Original -> BAF -> Hot-Pixel -> Isolated-Event -> Voxel Grid, with %
reduction at every stage and in-box/out-of-box removal breakdown), the
computational complexity analysis for every preprocessing module, saves a
before/after visualization, and writes M4_report.json / M4_report.md.

Usage:
    python generate_m4_report.py --dataset_root /path/to/dataset --output_dir /content/m4_report
================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from preprocessing_config import PreprocessingConfig
from event_preprocessor import EventPreprocessor
from preprocessing_visualization import visualize_preprocessing_stages
from voxel_grid import VoxelGridConfig, VoxelGridGenerator, estimate_voxelization_cost
from background_activity_filter import estimate_baf_cost
from isolated_event_filter import estimate_isolated_filter_cost
from hot_pixel_filter import estimate_hot_pixel_detection_cost
from test_m4_preprocessing import run_all_m4_tests


def generate_m4_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    num_sample_windows: int = 30,
    config: Optional[PreprocessingConfig] = None,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    config = config or PreprocessingConfig()

    all_passed, test_results = run_all_m4_tests(dataset_root, window_us=window_us)

    manifest = create_split(dataset_root, seed=seed)
    dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
    preprocessor = EventPreprocessor(config)
    voxel_config = VoxelGridConfig()
    voxel_gen = VoxelGridGenerator(voxel_config)

    n_sample = min(num_sample_windows, len(dataset))

    # ---- Aggregate the preprocessing ledger across sampled windows ----
    stage_totals = {}   # stage_name -> {"n_before":sum, "n_after":sum, "n_removed":sum,
                         #                "n_removed_inside":sum, "n_removed_outside":sum, "time":sum}
    memory_bytes_samples = []
    example_result = None
    example_sample = None

    for i in range(n_sample):
        sample = dataset[i]
        n_events_before_mem = sample["t"].numel() * 4 * 4  # 4 arrays, int32-ish, rough estimate
        result = preprocessor.process(sample, dataset_root=dataset_root,
                                       capture_snapshots=(example_result is None and len(sample["boxes"]) > 0))
        if example_result is None:
            example_result = result
            example_sample = sample

        memory_bytes_samples.append(n_events_before_mem)

        for s in result["stage_stats"]:
            if s.stage not in stage_totals:
                stage_totals[s.stage] = {
                    "n_before": 0, "n_after": 0, "n_removed": 0,
                    "n_removed_inside": 0, "n_removed_outside": 0, "time": 0.0,
                }
            st = stage_totals[s.stage]
            st["n_before"] += s.n_before
            st["n_after"] += s.n_after
            st["n_removed"] += s.n_removed
            st["n_removed_inside"] += (s.n_removed_inside_box or 0)
            st["n_removed_outside"] += (s.n_removed_outside_box or 0)
            st["time"] += s.elapsed_time_s

    if example_result is None or "snapshots" not in example_result:
        # fall back: force a visualization even if no box-containing window was found
        example_sample = dataset[0]
        example_result = preprocessor.process(example_sample, dataset_root=dataset_root, capture_snapshots=True)

    viz_path = os.path.join(output_dir, "preprocessing_stages_example.png")
    visualize_preprocessing_stages(
        example_result["snapshots"], example_sample["sensor_height"], example_sample["sensor_width"],
        boxes=example_sample["boxes"], out_path=viz_path,
    )

    # ---- Build the requested pipeline ledger with cumulative % reduction ----
    ledger = []
    n0 = stage_totals.get("original", {}).get("n_before", 0)
    stage_order = ["original", "baf", "hot_pixel_filter", "isolated_event_filter"]
    running = n0
    for stage in stage_order:
        if stage not in stage_totals:
            continue
        st = stage_totals[stage]
        running = st["n_after"]
        ledger.append({
            "stage": stage,
            "n_before": st["n_before"],
            "n_after": st["n_after"],
            "n_removed": st["n_removed"],
            "pct_removed_this_stage": (100.0 * st["n_removed"] / st["n_before"]) if st["n_before"] else 0.0,
            "pct_removed_cumulative": (100.0 * (n0 - running) / n0) if n0 else 0.0,
            "n_removed_inside_box": st["n_removed_inside"],
            "n_removed_outside_box": st["n_removed_outside"],
            "pct_removed_inside_box": (100.0 * st["n_removed_inside"] / st["n_removed"]) if st["n_removed"] else 0.0,
            "total_stage_time_s": st["time"],
        })
    ledger.append({
        "stage": "voxel_grid",
        "note": f"Events aggregated into a fixed-size ({voxel_config.num_channels}, H, W) tensor -- "
                f"'event count' no longer applies past this point. See M3 cost ledger for output size.",
    })
    ledger.append({
        "stage": "patch_generation",
        "note": "Not yet implemented (M5). This ledger will be extended once M5 exists.",
    })

    # ---- Computational complexity analysis (item 9) ----
    avg_events_per_window = (n0 / n_sample) if n_sample else 0
    h, w = example_sample["sensor_height"], example_sample["sensor_width"]
    complexity_analysis = {
        "baf": estimate_baf_cost(int(avg_events_per_window), config, h, w),
        "isolated_event_filter": estimate_isolated_filter_cost(int(avg_events_per_window), config, h, w),
        "hot_pixel_detection": estimate_hot_pixel_detection_cost(
            int(avg_events_per_window * len(dataset.stems) * 10), h, w  # rough full-recording scale note
        ),
        "voxelization_downstream_reference": estimate_voxelization_cost(int(avg_events_per_window), voxel_config, h, w),
    }

    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M4 - Noise Reduction & Event Preprocessing",
        "config": config.__dict__,
        "num_windows_sampled": n_sample,
        "preprocessing_ledger": ledger,
        "computational_complexity": complexity_analysis,
        "example_visualization": viz_path,
        "avg_memory_bytes_per_window_raw_estimate": float(np.mean(memory_bytes_samples)) if memory_bytes_samples else 0.0,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
    }

    json_path = os.path.join(output_dir, "M4_report.json")
    md_path = os.path.join(output_dir, "M4_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M4 - Noise Reduction & Event Preprocessing Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Preprocessing Ledger (aggregated over {n_sample} sampled windows)\n\n")
        f.write("| Stage | Before | After | Removed | % Removed (stage) | % Removed (cumulative) | % of removed that was in-box |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for entry in ledger:
            if "n_before" in entry:
                f.write(f"| {entry['stage']} | {entry['n_before']} | {entry['n_after']} | "
                        f"{entry['n_removed']} | {entry['pct_removed_this_stage']:.2f}% | "
                        f"{entry['pct_removed_cumulative']:.2f}% | {entry['pct_removed_inside_box']:.2f}% |\n")
            else:
                f.write(f"| {entry['stage']} | — | — | — | — | — | {entry['note']} |\n")

        f.write(
            "\n_EDPS-safety check: 'in-box' % should be LOW and comparable to (or lower than) the "
            "background base rate -- if a filter removes a much higher share of in-box events than "
            "background, it is behaving as semantic/object filtering rather than noise removal, and "
            "its thresholds should be loosened._\n"
        )

        f.write("\n## Computational Complexity Analysis\n\n")
        for module_name, cost in complexity_analysis.items():
            f.write(f"### {module_name}\n")
            for k, v in cost.items():
                f.write(f"- **{k}**: {v}\n")
            f.write("\n")

        f.write(f"## Example Visualization\n\n- `{viz_path}`\n")

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
    ap.add_argument("--output_dir", default="./m4_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--num_sample_windows", type=int, default=30)
    args = ap.parse_args()

    generate_m4_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        num_sample_windows=args.num_sample_windows,
    )
