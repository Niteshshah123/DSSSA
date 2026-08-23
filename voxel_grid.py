from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch


@dataclass
class VoxelGridConfig:
    num_bins: int = 10
    temporal_mode: str = "bilinear"
    polarity_encoding: str = "signed"

    def __post_init__(self):
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {self.num_bins}")
        if self.temporal_mode not in ("bilinear", "count"):
            raise ValueError(
                f"temporal_mode must be 'bilinear' or 'count', got {self.temporal_mode!r}"
            )
        if self.polarity_encoding not in ("signed", "separate_channels"):
            raise ValueError(
                "polarity_encoding must be 'signed' or 'separate_channels', "
                f"got {self.polarity_encoding!r}"
            )

    @property
    def num_channels(self) -> int:
        return self.num_bins if self.polarity_encoding == "signed" else 2 * self.num_bins


def events_to_voxel_grid(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t_start: int,
    t_end: int,
    height: int,
    width: int,
    config: VoxelGridConfig,
) -> np.ndarray:
    """
    Convert one event window to a float32 voxel grid.

    Semantics are unchanged:
      - bilinear temporal interpolation or count binning
      - signed or separate polarity channels
    """
    grid = np.zeros(
        (config.num_channels, height, width), dtype=np.float32
    )

    n_events = len(t)
    if n_events == 0:
        return grid

    duration = t_end - t_start
    if duration <= 0:
        raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start})")

    # Keep the computation in float32. The original implementation promoted
    # the entire temporal-coordinate path to float64, which adds CPU work and
    # memory traffic without improving the final float32 grid.
    norm_t = (
        (np.asarray(t, dtype=np.float32) - np.float32(t_start))
        / np.float32(duration)
        * np.float32(config.num_bins - 1)
    )
    np.clip(norm_t, 0.0, config.num_bins - 1, out=norm_t)

    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)

    if config.polarity_encoding == "signed":
        pol = np.where(np.asarray(p) > 0, 1.0, -1.0).astype(np.float32, copy=False)
        _splat(grid, norm_t, y, x, pol, config, channel_offset=0)
    else:
        on_mask = np.asarray(p) > 0
        off_mask = ~on_mask

        if np.any(on_mask):
            _splat(
                grid,
                norm_t[on_mask],
                y[on_mask],
                x[on_mask],
                np.ones(np.count_nonzero(on_mask), dtype=np.float32),
                config,
                channel_offset=0,
            )

        if np.any(off_mask):
            _splat(
                grid,
                norm_t[off_mask],
                y[off_mask],
                x[off_mask],
                np.ones(np.count_nonzero(off_mask), dtype=np.float32),
                config,
                channel_offset=config.num_bins,
            )

    return grid


def _splat(
    grid: np.ndarray,
    norm_t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
    config: VoxelGridConfig,
    channel_offset: int,
) -> None:
    """In-place temporal accumulation with the same semantics as M3."""
    num_bins = config.num_bins
    _, h, w = grid.shape
    spatial_size = h * w

    if config.temporal_mode == "count":
        bin_idx = np.floor(norm_t).astype(np.int64)
        np.clip(bin_idx, 0, num_bins - 1, out=bin_idx)
        flat_idx = (
            (channel_offset + bin_idx) * spatial_size + y * w + x
        )
        counts = np.bincount(
            flat_idx,
            weights=weight,
            minlength=grid.size,
        )
        grid += counts.reshape(grid.shape).astype(np.float32, copy=False)
        return

    # Bilinear temporal interpolation.
    t0 = np.floor(norm_t).astype(np.int64)
    t1 = np.minimum(t0 + 1, num_bins - 1)
    frac = norm_t - t0

    w0 = 1.0 - frac
    w1 = frac

    flat_idx0 = (
        (channel_offset + t0) * spatial_size + y * w + x
    )
    vals0 = np.bincount(
        flat_idx0,
        weights=weight * w0,
        minlength=grid.size,
    )
    grid += vals0.reshape(grid.shape).astype(np.float32, copy=False)

    distinct = t1 != t0
    if np.any(distinct):
        flat_idx1 = (
            (channel_offset + t1[distinct]) * spatial_size
            + y[distinct] * w
            + x[distinct]
        )
        vals1 = np.bincount(
            flat_idx1,
            weights=weight[distinct] * w1[distinct],
            minlength=grid.size,
        )
        grid += vals1.reshape(grid.shape).astype(np.float32, copy=False)


class VoxelGridGenerator:
    def __init__(self, config: VoxelGridConfig = None):
        self.config = config or VoxelGridConfig()

    def generate(self, sample: Dict[str, Any]) -> torch.Tensor:
        # Dataset tensors are CPU tensors. np.asarray/NumPy conversion here
        # avoids unnecessary copies when the source is already contiguous.
        def as_numpy(v):
            if torch.is_tensor(v):
                return v.detach().cpu().numpy()
            return np.asarray(v)

        grid = events_to_voxel_grid(
            as_numpy(sample["t"]),
            as_numpy(sample["x"]),
            as_numpy(sample["y"]),
            as_numpy(sample["p"]),
            t_start=sample["t_start"],
            t_end=sample["t_end"],
            height=sample["sensor_height"],
            width=sample["sensor_width"],
            config=self.config,
        )
        return torch.from_numpy(grid)


def estimate_voxelization_cost(
    n_events: int, config: VoxelGridConfig, height: int, width: int
) -> Dict[str, Any]:
    ops_per_event = 4 if config.temporal_mode == "bilinear" else 2
    estimated_flops = n_events * ops_per_event
    output_shape = (config.num_channels, height, width)
    output_elements = config.num_channels * height * width
    output_bytes = output_elements * 4

    return {
        "time_complexity": "O(N_events)",
        "space_complexity": "O(num_channels * H * W), independent of N_events",
        "n_events": n_events,
        "estimated_flops": estimated_flops,
        "output_shape": output_shape,
        "output_elements": output_elements,
        "output_bytes": output_bytes,
    }
