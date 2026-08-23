"""
================================================================================
M2 Per-Split Dataset Statistics (M2, item 7)
================================================================================
Computes statistics over an already-constructed EDPSGen1Dataset (i.e. after
splitting + windowing), to confirm the split is reasonably balanced and to
report window-level dataset composition for the paper.
================================================================================
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Any

import numpy as np

from gen1_dataset import EDPSGen1Dataset


def compute_split_statistics(
    dataset: EDPSGen1Dataset,
    max_windows: int = 2000,
) -> Dict[str, Any]:

    n_windows = len(dataset)
    n_recordings = len(dataset.stems)

    events_per_window = []
    boxes_per_window = []
    class_counts = Counter()

    # Decide which windows to inspect
    if n_windows <= max_windows:
        indices = range(n_windows)
    else:
        # Uniformly sample windows across the dataset
        indices = np.linspace(
            0,
            n_windows - 1,
            max_windows,
            dtype=int
        )

    # Compute statistics only on selected windows
    for i in indices:
        sample = dataset[i]

        events_per_window.append(len(sample["t"]))
        boxes_per_window.append(len(sample["boxes"]))

        if len(sample["boxes"]) > 0:
            class_counts.update(sample["boxes"]["class_id"].tolist())

    events_per_window = np.array(events_per_window)
    boxes_per_window = np.array(boxes_per_window)

    return {
        "split": dataset.split,
        "num_recordings": n_recordings,
        "num_windows": n_windows,
        "window_us": dataset.window_us,
        "stride_us": dataset.stride_us,
        "avg_events_per_window": float(np.mean(events_per_window)) if len(events_per_window) else 0.0,
        "std_events_per_window": float(np.std(events_per_window)) if len(events_per_window) else 0.0,
        "avg_boxes_per_window": float(np.mean(boxes_per_window)) if len(boxes_per_window) else 0.0,
        "std_boxes_per_window": float(np.std(boxes_per_window)) if len(boxes_per_window) else 0.0,
        "empty_windows": int(np.sum(boxes_per_window == 0)),
        "empty_window_fraction": float(np.mean(boxes_per_window == 0)) if len(boxes_per_window) else 0.0,
        "class_distribution": {int(k): int(v) for k, v in class_counts.items()},
    }
