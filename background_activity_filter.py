"""
================================================================================
Background Activity Filter (M4, item 1)
================================================================================
See neighbor_support.py for the underlying algorithm. This module is just
the BAF-specific configuration/entry-point wrapper, kept separate so BAF can
be toggled and ablated independently of isolated-event removal even though
both share the same primitive.
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from neighbor_support import compute_neighbor_support
from preprocessing_config import PreprocessingConfig


def apply_baf(
    t: np.ndarray, x: np.ndarray, y: np.ndarray,
    height: int, width: int, config: PreprocessingConfig,
) -> np.ndarray:
    """Returns a boolean keep-mask, same length as t."""
    support = compute_neighbor_support(
        t, x, y, height, width,
        radius=config.baf_radius, time_window_us=config.baf_time_window_us,
    )
    return support >= config.baf_min_neighbors


def estimate_baf_cost(n_events: int, config: PreprocessingConfig, height: int, width: int) -> Dict[str, Any]:
    k = (2 * config.baf_radius + 1) ** 2 - 1
    return {
        "time_complexity": "O(N_events * K), K = (2*radius+1)^2 - 1",
        "space_complexity": "O(height * width) for the time-surface grid",
        "n_events": n_events,
        "k_neighbors_checked": k,
        "estimated_flops": n_events * k,  # one comparison + counter increment per neighbor check
        "time_surface_bytes": height * width * 8,  # float64 grid
    }
