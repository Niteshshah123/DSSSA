"""
================================================================================
M1 Extended Dataset Statistics
================================================================================
Computes the additional dataset-level statistics requested for the research
paper's dataset section:

  - Average bounding boxes per recording
  - Bounding box size distribution (area = w*h, in pixels^2)
  - Bounding box aspect ratio distribution (w/h)
  - Event density heatmap (spatial, across the sensor plane)
  - Event density distribution (temporal: per-recording event rate; and
    spatial: distribution of per-pixel counts within the heatmap)
  - Average events per bounding box
  - Average events outside bounding boxes

DOCUMENTED METHODOLOGY / ASSUMPTIONS (state these in the paper's methodology
section, since they affect the numbers):

  1. "Events per bounding box" / "events outside bounding boxes" require a
     TEMPORAL WINDOW around each box's single annotation timestamp, since a
     box represents an instant, not a duration. We use a symmetric window
     of `window_us` (default 50,000 us = 50 ms, matching common event-frame
     accumulation windows used in event-based detection literature) centered
     on the box's timestamp: [ts - window_us/2, ts + window_us/2).

  2. Within that window, an event is "inside" the box if its (x, y) falls in
     the box's rectangle, and "outside" otherwise. If two boxes' windows
     overlap in time, an event can be counted under more than one box's
     window -- this is a documented approximation for a per-box AGGREGATE
     statistic (not a per-frame partition), and is reported as such.

  3. Event density heatmap: streamed and accumulated in chunks directly from
     disk (never holding a full file's events in memory at once), so it
     scales to the largest recordings without risking Colab RAM limits.

  4. Per-bbox event counting requires full per-file (t, x, y) arrays (to
     mask by both time and space simultaneously). This is done ONE FILE AT
     A TIME, discarding arrays before moving to the next file, so peak
     memory is bounded by the single largest recording rather than the
     whole dataset. Can be restricted to a subset via `max_files` for quick
     iteration.
================================================================================
"""

from __future__ import annotations

import os
import struct
from typing import List, Optional, Dict, Any

import numpy as np

from event_parser import EventParser, PropheseeGen1Reader


# ==============================================================================
# 1-2. BOUNDING BOX GEOMETRY STATISTICS
# ==============================================================================
def compute_bbox_geometry_stats(dataset_root: str, stems: List[str]) -> Dict[str, Any]:
    """
    Returns per-recording box counts, and flat arrays of box area / aspect
    ratio across the whole dataset (all recordings in `stems`).
    """
    parser = EventParser()
    boxes_per_recording = []
    all_areas = []
    all_aspect_ratios = []
    all_w = []
    all_h = []

    for stem in stems:
        bbox_path = os.path.join(dataset_root, stem + "_bbox.npy")
        boxes = parser.load_annotations(bbox_path)
        boxes_per_recording.append(len(boxes))
        w = boxes["w"].astype(np.float64)
        h = boxes["h"].astype(np.float64)
        safe_h = np.where(h == 0, np.nan, h)
        all_w.append(w)
        all_h.append(h)
        all_areas.append(w * h)
        all_aspect_ratios.append(w / safe_h)

    boxes_per_recording = np.array(boxes_per_recording)
    all_areas = np.concatenate(all_areas) if all_areas else np.array([])
    all_aspect_ratios = np.concatenate(all_aspect_ratios) if all_aspect_ratios else np.array([])
    all_w = np.concatenate(all_w) if all_w else np.array([])
    all_h = np.concatenate(all_h) if all_h else np.array([])

    def summarize(arr):
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return {}
        return {
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "mean": float(np.mean(arr)), "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90)),
        }

    return {
        "num_recordings": len(stems),
        "avg_boxes_per_recording": float(np.mean(boxes_per_recording)) if len(boxes_per_recording) else 0.0,
        "std_boxes_per_recording": float(np.std(boxes_per_recording)) if len(boxes_per_recording) else 0.0,
        "boxes_per_recording_raw": boxes_per_recording.tolist(),
        "box_area_summary": summarize(all_areas),
        "box_aspect_ratio_summary": summarize(all_aspect_ratios),
        "box_width_summary": summarize(all_w),
        "box_height_summary": summarize(all_h),
        "_raw_areas": all_areas,          # kept for plotting; strip before JSON dump
        "_raw_aspect_ratios": all_aspect_ratios,
    }


# ==============================================================================
# 3-4. EVENT DENSITY HEATMAP (streamed, memory-safe) + PER-RECORDING EVENT RATE
# ==============================================================================
def compute_event_density_heatmap(
    dataset_root: str,
    stems: List[str],
    chunk_events: int = 5_000_000,
    max_files: Optional[int] = None,
) -> np.ndarray:
    """
    Streams every recording's event stream in fixed-size chunks directly from
    disk and accumulates a (height, width) count histogram. Never holds more
    than `chunk_events` events in memory at once, regardless of file size.
    """
    reader = PropheseeGen1Reader()
    used_stems = stems[:max_files] if max_files else stems

    heatmap = None
    for stem in used_stems:
        dat_path = os.path.join(dataset_root, stem + "_td.dat")
        header_info = reader.parse_header(dat_path)
        h, w = header_info["height"], header_info["width"]
        if heatmap is None:
            heatmap = np.zeros((h, w), dtype=np.int64)
        elif heatmap.shape != (h, w):
            raise ValueError(
                f"{stem}: sensor resolution ({h}x{w}) differs from other "
                f"recordings ({heatmap.shape}) -- cannot accumulate into one "
                f"heatmap. Handle mixed-resolution datasets separately."
            )

        with open(dat_path, "rb") as f:
            f.seek(header_info["data_start_offset"])
            while True:
                raw = np.fromfile(f, dtype=reader.EVENT_RECORD_DTYPE, count=chunk_events)
                if raw.size == 0:
                    break
                x = (raw["_"] & reader.X_MASK).astype(np.int64)
                y = ((raw["_"] >> reader.Y_SHIFT) & reader.Y_MASK).astype(np.int64)
                np.add.at(heatmap, (y, x), 1)
                if raw.size < chunk_events:
                    break

    return heatmap if heatmap is not None else np.zeros((0, 0), dtype=np.int64)


def compute_event_rate_per_recording(dataset_root: str, stems: List[str]) -> List[Dict[str, Any]]:
    """
    Efficiently computes (event_count, duration, rate) per recording WITHOUT
    decoding the full file: reads only the header, the total byte count
    (-> event count), and the first/last 8-byte event records directly via
    targeted seeks (-> duration).
    """
    reader = PropheseeGen1Reader()
    results = []
    for stem in stems:
        dat_path = os.path.join(dataset_root, stem + "_td.dat")
        header_info = reader.parse_header(dat_path)
        t_min, t_max, event_count = reader.get_time_range(dat_path, header_info)

        duration_us = t_max - t_min
        duration_s = duration_us / 1e6 if duration_us > 0 else 0.0
        rate_hz = event_count / duration_s if duration_s > 0 else 0.0

        results.append({
            "stem": stem,
            "event_count": int(event_count),
            "duration_s": duration_s,
            "rate_hz": rate_hz,
        })
    return results


# ==============================================================================
# 5-6. AVERAGE EVENTS INSIDE / OUTSIDE BOUNDING BOXES
# ==============================================================================
def compute_bbox_event_counts(
    dataset_root: str,
    stems: List[str],
    window_us: int = 50_000,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """
    For each box, counts events falling inside its [ts-window/2, ts+window/2)
    time window AND inside its spatial rectangle ("inside"), vs. in the same
    time window but outside the rectangle ("outside"). See module docstring
    for the documented methodology/approximation.

    Processes one recording at a time, discarding its event arrays before
    moving to the next, so peak memory is bounded by the single largest file.
    """
    parser = EventParser()
    used_stems = stems[:max_files] if max_files else stems

    total_inside = 0
    total_outside = 0
    total_boxes = 0
    per_box_inside_counts = []

    for stem in used_stems:
        dat_path = os.path.join(dataset_root, stem + "_td.dat")
        bbox_path = os.path.join(dataset_root, stem + "_bbox.npy")

        events = parser.load_events(dat_path, validate=False)
        boxes = parser.load_annotations(bbox_path)
        t, x, y = events["t"], events["x"], events["y"]

        for b in boxes:
            ts = int(b["ts"])
            t_lo, t_hi = ts - window_us // 2, ts + window_us // 2
            time_mask = (t >= t_lo) & (t < t_hi)
            if not np.any(time_mask):
                per_box_inside_counts.append(0)
                total_boxes += 1
                continue

            xw, yw = x[time_mask], y[time_mask]
            x0, y0, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            inside_mask = (xw >= x0) & (xw < x0 + bw) & (yw >= y0) & (yw < y0 + bh)

            n_inside = int(np.sum(inside_mask))
            n_outside = int(np.sum(~inside_mask))

            total_inside += n_inside
            total_outside += n_outside
            total_boxes += 1
            per_box_inside_counts.append(n_inside)

        del events, t, x, y  # free before next (potentially large) file

    return {
        "window_us": window_us,
        "num_files_used": len(used_stems),
        "num_boxes_used": total_boxes,
        "avg_events_per_bbox": (total_inside / total_boxes) if total_boxes else 0.0,
        "avg_events_outside_bbox_per_window": (total_outside / total_boxes) if total_boxes else 0.0,
        "total_events_inside": total_inside,
        "total_events_outside": total_outside,
        "_raw_per_box_inside_counts": per_box_inside_counts,
    }


# ==============================================================================
# PLOTTING
# ==============================================================================
def save_statistics_plots(
    bbox_stats: Dict[str, Any],
    heatmap: np.ndarray,
    rate_stats: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Bounding box size (area) distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bbox_stats["_raw_areas"], bins=40, color="steelblue")
    ax.set_xlabel("Box area (pixels^2)")
    ax.set_ylabel("Count")
    ax.set_title("Bounding Box Size Distribution")
    p = os.path.join(output_dir, "bbox_size_distribution.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths["bbox_size_distribution"] = p

    # Aspect ratio distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(bbox_stats["_raw_aspect_ratios"], bins=40, color="darkorange")
    ax.set_xlabel("Aspect ratio (w / h)")
    ax.set_ylabel("Count")
    ax.set_title("Bounding Box Aspect Ratio Distribution")
    p = os.path.join(output_dir, "bbox_aspect_ratio_distribution.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths["bbox_aspect_ratio_distribution"] = p

    # Event density heatmap
    fig, ax = plt.subplots(figsize=(8, 8 * heatmap.shape[0] / max(heatmap.shape[1], 1)))
    im = ax.imshow(heatmap, cmap="inferno")
    ax.set_title("Event Density Heatmap (raw event counts per pixel)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    p = os.path.join(output_dir, "event_density_heatmap.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths["event_density_heatmap"] = p

    # Spatial density distribution (histogram of per-pixel counts in the heatmap)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(heatmap.flatten(), bins=50, color="seagreen")
    ax.set_xlabel("Events per pixel (across sampled recordings)")
    ax.set_ylabel("Number of pixels")
    ax.set_title("Spatial Event Density Distribution")
    ax.set_yscale("log")
    p = os.path.join(output_dir, "event_density_distribution_spatial.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths["event_density_distribution_spatial"] = p

    # Temporal density distribution (per-recording event rate)
    rates = [r["rate_hz"] for r in rate_stats]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(rates, bins=min(20, max(3, len(rates))), color="mediumpurple")
    ax.set_xlabel("Event rate (events/sec)")
    ax.set_ylabel("Number of recordings")
    ax.set_title("Temporal Event Density Distribution (per-recording rate)")
    p = os.path.join(output_dir, "event_density_distribution_temporal.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    paths["event_density_distribution_temporal"] = p

    return paths
