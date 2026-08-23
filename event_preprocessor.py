"""
================================================================================
EventPreprocessor (M4, item 5 - Modular Pipeline)
================================================================================
Chains the M4 stages in the requested order:

    Raw Events -> BAF -> Hot-Pixel/Row Filter -> Isolated-Event Removal
               -> Normalization (adds fields, removes nothing)

Each stage is independently toggleable via PreprocessingConfig. For every
removal stage, records:
  - n_before, n_after, n_removed, pct_removed
  - n_removed_inside_boxes, n_removed_outside_boxes (the EDPS-safety check:
    see module docstring in the design-review discussion -- if a "noise
    filter" disproportionately removes in-box events relative to background,
    that's a signal it's acting as semantic filtering, not noise removal)
  - elapsed_time_s

Hot-pixel detection is recording-level and cached (computed once per
recording, reused across every window of that recording), mirroring the
caching pattern already used in gen1_dataset.EDPSGen1Dataset.
================================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from preprocessing_config import PreprocessingConfig
from background_activity_filter import apply_baf
from isolated_event_filter import apply_isolated_event_filter
from hot_pixel_filter import detect_hot_pixels_for_recording, apply_hot_pixel_filter, HotPixelMask
from event_normalization import normalize_events
from event_parser import PropheseeGen1Reader


@dataclass
class StageStats:
    stage: str
    n_before: int
    n_after: int
    n_removed: int
    pct_removed: float
    n_removed_inside_box: Optional[int] = None
    n_removed_outside_box: Optional[int] = None
    pct_removed_inside_box: Optional[float] = None
    pct_removed_outside_box: Optional[float] = None
    elapsed_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _to_numpy(a):
    return a.numpy() if torch.is_tensor(a) else np.asarray(a)


def _inside_any_box(x: np.ndarray, y: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Bool mask, True where (x[i], y[i]) falls inside ANY box's rectangle."""
    if boxes is None or len(boxes) == 0 or len(x) == 0:
        return np.zeros(len(x), dtype=bool)
    inside = np.zeros(len(x), dtype=bool)
    for b in boxes:
        x0, y0, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
        inside |= (x >= x0) & (x < x0 + bw) & (y >= y0) & (y < y0 + bh)
    return inside


class EventPreprocessor:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.reader = PropheseeGen1Reader()
        self._hot_pixel_cache: Dict[str, HotPixelMask] = {}

    def _get_hot_pixel_mask(self, dataset_root: str, recording_stem: str) -> HotPixelMask:
        if recording_stem not in self._hot_pixel_cache:
            dat_path = os.path.join(dataset_root, recording_stem + "_td.dat")
            self._hot_pixel_cache[recording_stem] = detect_hot_pixels_for_recording(
                dat_path, self.reader, self.config
            )
        return self._hot_pixel_cache[recording_stem]

    def process(self, sample: Dict[str, Any], dataset_root: str, capture_snapshots: bool = False) -> Dict[str, Any]:
        """
        `sample` is an EDPSGen1Dataset-style dict (t/x/y/p as torch tensors,
        boxes, recording_stem, sensor_height/width, t_start/t_end).

        Returns a dict:
            {"t", "x", "y", "p": filtered int arrays (numpy),
             "normalized": dict of extra normalized fields (if enabled),
             "boxes": unchanged (M4 never touches annotations),
             "stage_stats": list[StageStats],
             "snapshots": list of {"name","x","y","p"} per stage, IF
                          capture_snapshots=True (used for item 6 visualization),
             ... (other passthrough metadata)}
        """
        t = _to_numpy(sample["t"]).astype(np.int64)
        x = _to_numpy(sample["x"]).astype(np.int64)
        y = _to_numpy(sample["y"]).astype(np.int64)
        p = _to_numpy(sample["p"]).astype(np.int64)
        boxes = sample["boxes"]
        height, width = sample["sensor_height"], sample["sensor_width"]

        # ------------------------------------------------------------
        # DEFENSIVE CHRONOLOGICAL SORT (M10 fix)
        # ------------------------------------------------------------
        # M4's BAF / isolated-event filters (via compute_neighbor_support)
        # have a STRICT precondition: ascending timestamps. Every upstream
        # stage that should already guarantee this (M1's parser, M2's
        # windowing, M10's augmentation) has been individually verified to
        # preserve it -- but a module with a strict precondition should
        # enforce it itself at its own boundary rather than trust every
        # possible caller to have done so perfectly, especially given a
        # failure was observed on real Colab data that this sandbox's
        # synthetic dataset never reproduced. This sort is an O(n log n)
        # no-op whenever the input is already sorted (which is the
        # expected common case) and a correctness guarantee otherwise.
        if len(t) > 0:
            sort_order = np.argsort(t, kind="stable")
            t, x, y, p = t[sort_order], x[sort_order], y[sort_order], p[sort_order]
            assert np.all(np.diff(t) >= 0), (
                "Internal error: events still not chronologically sorted after "
                "the defensive sort in EventPreprocessor.process() -- this "
                "should be mathematically impossible and indicates a bug in "
                "the sort itself, not in any upstream caller."
            )

        stage_stats: List[StageStats] = []
        snapshots: List[Dict[str, Any]] = []

        def _snapshot(name: str):
            if capture_snapshots:
                snapshots.append({"name": name, "x": x.copy(), "y": y.copy(), "p": p.copy()})

        def _run_removal_stage(name: str, keep_mask_fn):
            nonlocal t, x, y, p
            n_before = len(t)
            t0 = time.perf_counter()
            keep_mask = keep_mask_fn()
            elapsed = time.perf_counter() - t0

            removed_mask = ~keep_mask
            inside = _inside_any_box(x, y, boxes)
            n_removed_inside = int(np.sum(removed_mask & inside))
            n_removed_outside = int(np.sum(removed_mask & ~inside))

            t, x, y, p = t[keep_mask], x[keep_mask], y[keep_mask], p[keep_mask]
            n_after = len(t)
            n_removed = n_before - n_after

            stage_stats.append(StageStats(
                stage=name, n_before=n_before, n_after=n_after, n_removed=n_removed,
                pct_removed=(100.0 * n_removed / n_before) if n_before else 0.0,
                n_removed_inside_box=n_removed_inside,
                n_removed_outside_box=n_removed_outside,
                pct_removed_inside_box=(100.0 * n_removed_inside / n_removed) if n_removed else 0.0,
                pct_removed_outside_box=(100.0 * n_removed_outside / n_removed) if n_removed else 0.0,
                elapsed_time_s=elapsed,
            ))
            _snapshot(name)

        n_original = len(t)
        stage_stats.append(StageStats(
            stage="original", n_before=n_original, n_after=n_original, n_removed=0, pct_removed=0.0,
        ))
        _snapshot("original")

        if self.config.use_baf:
            _run_removal_stage("baf", lambda: apply_baf(t, x, y, height, width, self.config))

        if self.config.use_hot_pixel_filter:
            hot_mask = self._get_hot_pixel_mask(dataset_root, sample["recording_stem"])
            _run_removal_stage("hot_pixel_filter", lambda: apply_hot_pixel_filter(x, y, hot_mask))

        if self.config.use_isolated_event_filter:
            _run_removal_stage(
                "isolated_event_filter",
                lambda: apply_isolated_event_filter(t, x, y, height, width, self.config),
            )

        normalized = {}
        if self.config.use_normalization:
            normalized = normalize_events(
                t, x, y, p, sample["t_start"], sample["t_end"], height, width, self.config
            )

        result = {
            "t": t, "x": x, "y": y, "p": p,
            "normalized": normalized,
            "boxes": boxes,
            "recording_stem": sample["recording_stem"],
            "window_index": sample["window_index"],
            "t_start": sample["t_start"], "t_end": sample["t_end"],
            "sensor_height": height, "sensor_width": width,
            "stage_stats": stage_stats,
        }
        if capture_snapshots:
            result["snapshots"] = snapshots
        return result
