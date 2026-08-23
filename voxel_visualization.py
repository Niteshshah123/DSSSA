"""
================================================================================
Voxel Grid Visualization (M3)
================================================================================
Two views, since a voxel grid isn't directly viewable as one image:
  1. A per-bin montage (each temporal bin shown as its own small image),
     which is the most honest representation of what the tensor contains.
  2. A single "collapsed" view (sum/abs-sum across bins), for a quick
     at-a-glance sanity check against the same window's raw event render
     from M1/M2.
================================================================================
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def visualize_voxel_grid(
    voxel: torch.Tensor,
    config,
    out_path: str = "voxel_grid_visualization.png",
    boxes: Optional[np.ndarray] = None,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    grid = voxel.numpy() if torch.is_tensor(voxel) else np.asarray(voxel)
    n_channels, h, w = grid.shape

    if config.polarity_encoding == "signed":
        collapsed = grid.sum(axis=0)
        vmax = max(abs(collapsed.min()), abs(collapsed.max()), 1e-6)
        cmap, vmin, vmax_ = "bwr", -vmax, vmax
        per_bin_channels = list(range(n_channels))
    else:
        num_bins = config.num_bins
        collapsed = grid[:num_bins].sum(axis=0) - grid[num_bins:].sum(axis=0)
        vmax = max(abs(collapsed.min()), abs(collapsed.max()), 1e-6)
        cmap, vmin, vmax_ = "bwr", -vmax, vmax
        per_bin_channels = list(range(num_bins))  # show ON bins as representative

    n_show = len(per_bin_channels)
    ncols = min(5, n_show)
    nrows = math.ceil(n_show / ncols) + 1  # +1 row for the collapsed view

    fig = plt.figure(figsize=(3 * ncols, 3 * nrows))

    ax_collapsed = fig.add_subplot(nrows, 1, 1)
    im = ax_collapsed.imshow(collapsed, cmap=cmap, vmin=vmin, vmax=vmax_)
    ax_collapsed.set_title("Collapsed voxel grid (sum over bins)")
    ax_collapsed.axis("off")
    if boxes is not None:
        for b in boxes:
            rect = patches.Rectangle((b["x"], b["y"]), b["w"], b["h"],
                                      linewidth=1.2, edgecolor="lime", facecolor="none")
            ax_collapsed.add_patch(rect)
    fig.colorbar(im, ax=ax_collapsed, fraction=0.025)

    for i, ch in enumerate(per_bin_channels):
        ax = fig.add_subplot(nrows, ncols, ncols + i + 1)
        ax.imshow(grid[ch], cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(f"bin {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
