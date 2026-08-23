"""
================================================================================
M1 Research Log Report Generator
================================================================================
Runs the EventParser unit tests, computes both a single-sample-file statistics
snapshot AND the extended dataset-wide statistics (bbox geometry, event
density, events-in/out-of-bbox), and writes a report (JSON + Markdown)
summarizing:
  - parser configuration
  - sample-file statistics
  - extended dataset-wide statistics + saved plots
  - validation (unit test) results
  - execution time

The Markdown version is meant to be directly usable as source material for
the paper's methodology / dataset section later.

Usage (in Colab, after mounting Drive and placing all M1 .py files on the path):

    python generate_m1_report.py \
        --dataset_root /content/drive/MyDrive/extracted_dataset_from_colab/detection_dataset_duration_60s_ratio_1.0/test/ \
        --sample_stem 17-04-04_11-00-13_cut_15_122500000_182500000 \
        --output_dir /content/m1_report

By default this processes ALL recordings for the extended statistics. Use
--max_files_for_extended_stats to bound runtime while iterating.
================================================================================
"""

import os
import glob
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Optional

from event_parser import EventParser
from test_event_parser import run_all_tests
from m1_dataset_statistics import (
    compute_bbox_geometry_stats,
    compute_event_density_heatmap,
    compute_event_rate_per_recording,
    compute_bbox_event_counts,
    save_statistics_plots,
)


def _list_stems(dataset_root: str):
    dat_files = sorted(glob.glob(os.path.join(dataset_root, "*_td.dat")))
    return [os.path.basename(f)[: -len("_td.dat")] for f in dat_files]


def generate_m1_report(
    dataset_root: str,
    sample_stem: str,
    output_dir: str = ".",
    compute_extended_stats: bool = True,
    max_files_for_extended_stats: Optional[int] = None,
    bbox_event_window_us: int = 50_000,
    heatmap_chunk_events: int = 5_000_000,
):
    start_time = time.time()
    parser = EventParser()

    dat_path = os.path.join(dataset_root, sample_stem + "_td.dat")
    bbox_path = os.path.join(dataset_root, sample_stem + "_bbox.npy")

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Sample .dat file not found: {dat_path}")
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"Sample .bbox.npy file not found: {bbox_path}")

    # --- Unit tests (on the single sample file) ---
    all_passed, test_results = run_all_tests(dat_path, bbox_path)

    # --- Single-sample-file statistics ---
    events = parser.load_events(dat_path, validate=True)
    boxes = parser.load_annotations(bbox_path)
    sample_stats = parser.get_event_statistics(events, boxes=boxes)

    all_stems = _list_stems(dataset_root)
    dataset_level = {"num_recordings_found": len(all_stems)}

    # --- Extended dataset-wide statistics ---
    extended = None
    plot_paths = None
    if compute_extended_stats:
        used_stems = all_stems[:max_files_for_extended_stats] if max_files_for_extended_stats else all_stems
        print(f"\nComputing extended dataset-wide statistics over "
              f"{len(used_stems)} recording(s)...")

        bbox_geom = compute_bbox_geometry_stats(dataset_root, used_stems)
        heatmap = compute_event_density_heatmap(
            dataset_root, used_stems, chunk_events=heatmap_chunk_events
        )
        rate_stats = compute_event_rate_per_recording(dataset_root, used_stems)
        bbox_event_stats = compute_bbox_event_counts(
            dataset_root, used_stems, window_us=bbox_event_window_us
        )

        plots_dir = os.path.join(output_dir, "plots")
        plot_paths = save_statistics_plots(bbox_geom, heatmap, rate_stats, plots_dir)

        extended = {
            "num_recordings_used": len(used_stems),
            "avg_boxes_per_recording": bbox_geom["avg_boxes_per_recording"],
            "std_boxes_per_recording": bbox_geom["std_boxes_per_recording"],
            "box_area_summary": bbox_geom["box_area_summary"],
            "box_aspect_ratio_summary": bbox_geom["box_aspect_ratio_summary"],
            "box_width_summary": bbox_geom["box_width_summary"],
            "box_height_summary": bbox_geom["box_height_summary"],
            "avg_events_per_bbox": bbox_event_stats["avg_events_per_bbox"],
            "avg_events_outside_bbox_per_window": bbox_event_stats["avg_events_outside_bbox_per_window"],
            "bbox_event_window_us": bbox_event_stats["window_us"],
            "bbox_event_stats_num_files_used": bbox_event_stats["num_files_used"],
            "bbox_event_stats_num_boxes_used": bbox_event_stats["num_boxes_used"],
            "per_recording_event_rate_hz": [r["rate_hz"] for r in rate_stats],
        }
        print(f"Extended statistics computed over {len(used_stems)} recording(s).")

    elapsed = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M1 - EventParser",
        "parser_config": {
            "reader_class": type(parser.reader).__name__,
            "validate_on_load_default": parser.validate_on_load,
            "event_record_size_bytes": 8,
            "timestamp_units": "microseconds",
            "polarity_encoding": "0 = OFF (brightness decrease), 1 = ON (brightness increase)",
            "sensor_resolution_source": "auto-inferred from per-file ASCII header (Height/Width)",
            "bbox_event_association_window_us": bbox_event_window_us,
        },
        "sample_file": {"dat_path": dat_path, "bbox_path": bbox_path},
        "dataset_level": dataset_level,
        "sample_statistics": sample_stats.to_dict(),
        "extended_statistics": extended,
        "extended_statistics_plots": plot_paths,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed, 4),
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "M1_report.json")
    md_path = os.path.join(output_dir, "M1_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w") as f:
        f.write("# M1 - EventParser Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Parser Configuration\n\n")
        for k, v in report["parser_config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Dataset Context\n\n")
        f.write(f"- Recordings found in dataset root: {dataset_level['num_recordings_found']}\n")

        f.write(f"\n## Sample File Statistics (`{sample_stem}`)\n\n")
        for k, v in report["sample_statistics"].items():
            f.write(f"- **{k}**: {v}\n")

        if extended is not None:
            f.write(f"\n## Extended Dataset Statistics "
                    f"(across {extended['num_recordings_used']} recordings)\n\n")
            f.write(f"- **Average bounding boxes per recording**: "
                    f"{extended['avg_boxes_per_recording']:.2f} "
                    f"(std={extended['std_boxes_per_recording']:.2f})\n")
            f.write(f"- **Bounding box area (px^2)**: {extended['box_area_summary']}\n")
            f.write(f"- **Bounding box aspect ratio (w/h)**: {extended['box_aspect_ratio_summary']}\n")
            f.write(f"- **Bounding box width (px)**: {extended['box_width_summary']}\n")
            f.write(f"- **Bounding box height (px)**: {extended['box_height_summary']}\n")
            f.write(f"- **Average events per bounding box**: "
                    f"{extended['avg_events_per_bbox']:.2f} "
                    f"(window={extended['bbox_event_window_us']}us, "
                    f"{extended['bbox_event_stats_num_boxes_used']} boxes over "
                    f"{extended['bbox_event_stats_num_files_used']} files)\n")
            f.write(f"- **Average events outside bounding boxes per window**: "
                    f"{extended['avg_events_outside_bbox_per_window']:.2f}\n")

            if plot_paths:
                f.write("\n### Saved Plots\n\n")
                for name, p in plot_paths.items():
                    f.write(f"- {name}: `{p}`\n")

            f.write(
                "\n_Methodology note: 'events per bounding box' / 'events "
                "outside bounding boxes' use a symmetric time window centered "
                "on each box's annotation timestamp (window length above), "
                "since annotations mark an instant, not a duration. Overlapping "
                "box windows can double-count events; this is a documented "
                "aggregate approximation, not a per-frame partition._\n"
            )

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
    ap.add_argument(
        "--dataset_root",
        default="/content/drive/MyDrive/extracted_dataset_from_colab/detection_dataset_duration_60s_ratio_1.0/test/",
    )
    ap.add_argument(
        "--sample_stem",
        default="17-04-04_11-00-13_cut_15_122500000_182500000",
    )
    ap.add_argument("--output_dir", default="/content/m1_report")
    ap.add_argument("--max_files_for_extended_stats", type=int, default=None)
    ap.add_argument("--bbox_event_window_us", type=int, default=50_000)
    args = ap.parse_args()

    generate_m1_report(
        args.dataset_root,
        args.sample_stem,
        args.output_dir,
        max_files_for_extended_stats=args.max_files_for_extended_stats,
        bbox_event_window_us=args.bbox_event_window_us,
    )
