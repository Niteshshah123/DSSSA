"""
================================================================================
EDPSGen1Dataset (M2, item 4)
================================================================================
A torch.utils.data.Dataset over temporal windows of the Prophesee Gen1
dataset, for one split ("train"/"val"/"test") of a SplitManifest.

Each __getitem__ returns a dict of RAW, variable-length per-window event
data plus its associated ground-truth boxes:

    {
        "t": int64[N], "x": int32[N], "y": int32[N], "p": int8[N],
        "boxes": structured ndarray (may be length 0),
        "recording_stem": str, "window_index": int,
        "t_start": int, "t_end": int,
        "sensor_height": int, "sensor_width": int,
    }

Deliberately NOT voxelized/tokenized here -- that is the responsibility of
M3 (Voxel Grid Generation) and later modules, per the pipeline in the SRS.
Keeping this Dataset's output as raw (t, x, y, p) keeps M2 decoupled from
whatever voxelization/patching scheme M3 settles on.

CACHING DESIGN:
A single recording produces many windows. Re-opening and re-decoding the
same *_td.dat file for every window would be wasteful, so this class keeps
a small LRU cache of fully-decoded (t, x, y, p) arrays for the most
recently used recordings (`cache_size`, default 2). This works well when
consecutive __getitem__ calls tend to hit the same recording (e.g.
`shuffle=False`, or `torch.utils.data.DataLoader` with a moderate
`num_workers` where each worker iterates its own contiguous shard).

Under heavy random shuffling with many workers, cache hit rate will be
lower and repeated re-decoding will occur -- this is a known, documented
trade-off. A future optimization (not implemented here) would be to
pre-slice per-window byte offsets during manifest building so each window
can be read directly from disk without full-file decoding.
================================================================================
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from event_parser import EventParser, ParserValidationError
from dataset_split import SplitManifest
from temporal_windowing import (
    TemporalWindow,
    generate_windows_for_recording,
    assign_boxes_to_window,
)


def _find_dat_path(dataset_root: str, stem: str) -> str:
    p1 = os.path.join(dataset_root, stem + "_td.dat")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(dataset_root, stem + ".dat")
    if os.path.exists(p2):
        return p2
    return p1


class EDPSGen1Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        manifest: SplitManifest,
        split: str,
        window_us: int = 50_000,
        stride_us: Optional[int] = None,
        drop_empty_windows: bool = False,
        cache_size: int = 2,
        parser: Optional[EventParser] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train'/'val'/'test', got {split!r}")

        self.dataset_root = dataset_root
        self.split = split
        self.window_us = window_us
        self.stride_us = stride_us or window_us
        self.parser = parser or EventParser()
        self.cache_size = cache_size

        self.stems: List[str] = list(getattr(manifest, split))
        if len(self.stems) == 0:
            raise ValueError(
                f"Split '{split}' contains zero recordings in the given manifest."
            )

        self._event_cache: "OrderedDict[str, dict]" = OrderedDict()
        self._box_cache = {}

        self.windows: List[TemporalWindow] = []
        for stem in self.stems:
            dat_path = _find_dat_path(dataset_root, stem)
            header_info = self.parser.reader.parse_header(dat_path)
            t_min, t_max, _ = self.parser.reader.get_time_range(dat_path, header_info)

            wins = generate_windows_for_recording(
                stem, t_min, t_max, window_us=self.window_us, stride_us=self.stride_us
            )

            if drop_empty_windows:
                boxes = self._get_boxes(stem)
                wins = [w for w in wins if len(assign_boxes_to_window(boxes, w)) > 0]

            self.windows.extend(wins)

        if len(self.windows) == 0:
            raise ValueError(
                f"Split '{split}' produced zero windows (window_us={window_us} may be "
                f"longer than every recording's duration, or drop_empty_windows removed "
                f"all of them)."
            )

    def __len__(self) -> int:
        return len(self.windows)

    def _get_events(self, stem: str, t_start: int = None, t_end: int = None) -> dict:
        dat_path = _find_dat_path(self.dataset_root, stem)
        if t_start is not None and t_end is not None:
            return self.parser.load_events_window(
                dat_path, t_start=t_start, t_end=t_end, validate=False
            )

        if stem in self._event_cache:
            self._event_cache.move_to_end(stem)
            return self._event_cache[stem]

        events = self.parser.load_events(dat_path, validate=False)
        self._event_cache[stem] = events
        self._event_cache.move_to_end(stem)
        if len(self._event_cache) > self.cache_size:
            self._event_cache.popitem(last=False)
        return events

    def _get_boxes(self, stem: str) -> np.ndarray:
        if stem not in self._box_cache:
            bbox_path = os.path.join(self.dataset_root, stem + "_bbox.npy")
            if not os.path.exists(bbox_path):
                bbox_path = os.path.join(self.dataset_root, stem + ".npy")
            self._box_cache[stem] = self.parser.load_annotations(bbox_path)
        return self._box_cache[stem]

    def __getitem__(self, idx: int) -> dict:
        window = self.windows[idx]
        events = self._get_events(
            window.recording_stem, t_start=window.t_start, t_end=window.t_end
        )
        t, x, y, p = events["t"], events["x"], events["y"], events["p"]

        # The windowed reader already guarantees this range. Keep a defensive
        # mask so the Dataset remains correct for fallback/custom readers.
        mask = (t >= window.t_start) & (t < window.t_end)
        boxes = self._get_boxes(window.recording_stem)
        window_boxes = assign_boxes_to_window(boxes, window)

        return {
            "t": torch.from_numpy(t[mask].copy()),
            "x": torch.from_numpy(x[mask].copy()),
            "y": torch.from_numpy(y[mask].copy()),
            "p": torch.from_numpy(p[mask].copy()),
            "boxes": window_boxes,
            "recording_stem": window.recording_stem,
            "window_index": window.window_index,
            "t_start": window.t_start,
            "t_end": window.t_end,
            "sensor_height": events["header"]["height"],
            "sensor_width": events["header"]["width"],
        }
