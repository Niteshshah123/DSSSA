"""
================================================================================
Event Normalization (M4, item 4)
================================================================================
Produces ADDITIONAL normalized fields (t_norm, x_norm, y_norm, p_norm).
Does NOT replace the integer (t, x, y) arrays used for pixel indexing --
M3's VoxelGridGenerator continues to consume the filtered INTEGER
coordinates. This means toggling normalization on/off can never silently
change voxelization behavior; it only adds/removes extra fields that a
future module could use if it wants normalized inputs instead of a voxel
grid (e.g. a point-based/graph model, if the project ever explores that).
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from preprocessing_config import PreprocessingConfig


def normalize_events(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray,
    t_start: int, t_end: int, height: int, width: int,
    config: PreprocessingConfig,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if config.normalize_timestamp:
        duration = max(t_end - t_start, 1)
        result["t_norm"] = (t.astype(np.float64) - t_start) / duration
    if config.normalize_coordinates:
        result["x_norm"] = x.astype(np.float64) / max(width - 1, 1)
        result["y_norm"] = y.astype(np.float64) / max(height - 1, 1)
    if config.normalize_polarity:
        result["p_norm"] = np.where(p > 0, 1.0, -1.0)

    return result
