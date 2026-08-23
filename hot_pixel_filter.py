"""
================================================================================
Hot Pixel / Hot Row Suppression (M4, item 3)
================================================================================
DESIGN NOTE: a "hot pixel" or "hot row" is a SENSOR-LEVEL defect -- it's a
property of the recording (or the sensor itself), not of any single 50ms
window. Detecting it from a single window's event counts would be
statistically unreliable (too few samples per pixel). So detection happens
ONCE per recording, using the FULL recording's event stream (reusing the
same memory-safe streaming accumulation approach as M1's density heatmap),
and the resulting suppression mask is then applied identically to every
window drawn from that recording.

DETECTION METHOD: never hardcoded coordinates. Per-pixel and per-row event
counts are accumulated across the whole recording, then z-scored against
the count distribution (excluding near-empty pixels/rows, since a sparse
sensor region isn't what we're looking for -- only ABNORMALLY HIGH counts
matter). A pixel/row is flagged only if it is a statistically extreme
outlier (z-score above a configurable threshold, deliberately conservative
by default -- see PreprocessingConfig).

Every suppressed pixel/row is logged (coordinates + z-score + count) so the
decision is auditable in the M4 report, per the requirement to never make
this an invisible blanket policy.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

import numpy as np

from event_parser import PropheseeGen1Reader
from preprocessing_config import PreprocessingConfig


@dataclass
class HotPixelMask:
    height: int
    width: int
    hot_pixels: List[Tuple[int, int]] = field(default_factory=list)   # (y, x)
    hot_rows: List[int] = field(default_factory=list)
    pixel_mask: np.ndarray = None   # bool (H, W), True = suppress
    row_mask: np.ndarray = None     # bool (H,),   True = suppress
    log: List[Dict[str, Any]] = field(default_factory=list)           # auditable detail per flagged item

    def combined_pixel_mask(self) -> np.ndarray:
        """(H, W) bool mask: True wherever a pixel OR its row is flagged."""
        mask = self.pixel_mask.copy() if self.pixel_mask is not None else np.zeros((self.height, self.width), dtype=bool)
        if self.row_mask is not None:
            mask[self.row_mask, :] = True
        return mask


def detect_hot_pixels_for_recording(
    dat_path: str,
    reader: PropheseeGen1Reader,
    config: PreprocessingConfig,
    chunk_events: int = 5_000_000,
) -> HotPixelMask:
    """
    Streams the FULL recording once (memory-safe, chunked) to build a
    per-pixel event-count heatmap, then flags statistical outliers.
    """
    header_info = reader.parse_header(dat_path)
    h, w = header_info["height"], header_info["width"]
    heatmap = np.zeros((h, w), dtype=np.int64)

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

    result = HotPixelMask(height=h, width=w)
    result.pixel_mask = np.zeros((h, w), dtype=bool)
    result.row_mask = np.zeros(h, dtype=bool)

    if config.hot_pixel_enable_pixel_mode:
        flat = heatmap.flatten().astype(np.float64)
        considered = flat[flat >= config.hot_pixel_min_count]
        if len(considered) > 1 and considered.std() > 0:
            mean, std = considered.mean(), considered.std()
            z = (flat - mean) / std
            flagged_idx = np.where((flat >= config.hot_pixel_min_count) & (z > config.hot_pixel_z_thresh))[0]
            for idx in flagged_idx:
                yy, xx = divmod(int(idx), w)
                result.pixel_mask[yy, xx] = True
                result.hot_pixels.append((yy, xx))
                result.log.append({
                    "type": "pixel", "y": yy, "x": xx,
                    "count": int(heatmap[yy, xx]), "z_score": float(z[idx]),
                })

    if config.hot_pixel_enable_row_mode:
        row_counts = heatmap.sum(axis=1).astype(np.float64)
        considered = row_counts[row_counts >= config.hot_pixel_min_count]
        if len(considered) > 1 and considered.std() > 0:
            mean, std = considered.mean(), considered.std()
            z = (row_counts - mean) / std
            flagged_rows = np.where((row_counts >= config.hot_pixel_min_count) & (z > config.hot_row_z_thresh))[0]
            for yy in flagged_rows:
                result.row_mask[yy] = True
                result.hot_rows.append(int(yy))
                result.log.append({
                    "type": "row", "y": int(yy),
                    "count": int(row_counts[yy]), "z_score": float(z[yy]),
                })

    return result


def apply_hot_pixel_filter(x: np.ndarray, y: np.ndarray, hot_mask: HotPixelMask) -> np.ndarray:
    """Returns a boolean keep-mask (True = keep, event is NOT on a suppressed pixel/row)."""
    combined = hot_mask.combined_pixel_mask()
    if len(x) == 0:
        return np.ones(0, dtype=bool)
    return ~combined[y.astype(np.int64), x.astype(np.int64)]


def estimate_hot_pixel_detection_cost(n_events_full_recording: int, height: int, width: int) -> Dict[str, Any]:
    return {
        "time_complexity": "O(N_events_in_recording) for streaming heatmap accumulation, "
                            "+ O(height*width) for z-score outlier pass (once per recording)",
        "space_complexity": "O(height * width) for the heatmap, independent of event count",
        "n_events_full_recording": n_events_full_recording,
        "estimated_flops": n_events_full_recording + height * width * 3,
        "heatmap_bytes": height * width * 8,
    }
