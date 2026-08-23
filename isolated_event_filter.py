"""
================================================================================
Isolated Event Removal (M4, item 2)
================================================================================
Structurally identical mechanism to BAF (see neighbor_support.py), but a
SEPARATE pipeline stage with its own radius/window/threshold, applied to
whatever events survived the earlier stages (BAF, hot-pixel filtering, per
the requested pipeline order). Independently enabled/disabled via
`config.use_isolated_event_filter`.
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from neighbor_support import compute_neighbor_support
from preprocessing_config import PreprocessingConfig


def apply_isolated_event_filter(
    t: np.ndarray, x: np.ndarray, y: np.ndarray,
    height: int, width: int, config: PreprocessingConfig,
) -> np.ndarray:
    """Returns a boolean keep-mask, same length as t."""
    support = compute_neighbor_support(
        t, x, y, height, width,
        radius=config.iso_radius, time_window_us=config.iso_time_window_us,
    )
    return support >= config.iso_min_neighbors


def estimate_isolated_filter_cost(n_events: int, config: PreprocessingConfig, height: int, width: int) -> Dict[str, Any]:
    k = (2 * config.iso_radius + 1) ** 2 - 1
    return {
        "time_complexity": "O(N_events * K), K = (2*radius+1)^2 - 1",
        "space_complexity": "O(height * width) for the time-surface grid",
        "n_events": n_events,
        "k_neighbors_checked": k,
        "estimated_flops": n_events * k,
        "time_surface_bytes": height * width * 8,
    }
