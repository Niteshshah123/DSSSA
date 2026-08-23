"""
================================================================================
Patch Visualization (M5, item 5)
================================================================================
Two panels:
  1. Voxel grid (collapsed across bins) with patch boundaries drawn, patch
     indices numbered, and the top-K most active patches highlighted.
  2. A patch-level event-density heatmap (one value per patch, upsampled to
     patch-grid resolution) for a quick at-a-glance view of where the
     activity concentrates.
================================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

from patch_config import PatchConfig
from patch_metadata import PatchMetadata
from patch_statistics import PatchActivityStats
from voxel_grid import VoxelGridConfig


def visualize_patches(
    voxel: torch.Tensor,
    metadata: List[PatchMetadata],
    stats: List[PatchActivityStats],
    voxel_config: VoxelGridConfig,
    patch_config: PatchConfig,
    n_rows: int,
    n_cols: int,
    top_k_highlight: int = 5,
    out_path: str = "patch_visualization.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    grid = voxel.numpy() if torch.is_tensor(voxel) else np.asarray(voxel)
    if voxel_config.polarity_encoding == "signed":
        collapsed = grid.sum(axis=0)
    else:
        nb = voxel_config.num_bins
        collapsed = grid[:nb].sum(axis=0) - grid[nb:].sum(axis=0)
    h, w = collapsed.shape
    vmax = max(abs(collapsed.min()), abs(collapsed.max()), 1e-6)

    densities = np.array([s.event_density for s in stats])
    top_k_idx = set(np.argsort(densities)[-top_k_highlight:].tolist()) if len(densities) else set()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax0 = axes[0]
    ax0.imshow(collapsed, cmap="bwr", vmin=-vmax, vmax=vmax)
    ps = patch_config.patch_size
    for meta in metadata:
        edgecolor = "gold" if meta.patch_index in top_k_idx else "black"
        linewidth = 2.0 if meta.patch_index in top_k_idx else 0.4
        rect = mpatches.Rectangle(
            (meta.x0, meta.y0), meta.spatial_size, meta.spatial_size,
            linewidth=linewidth, edgecolor=edgecolor, facecolor="none",
        )
        ax0.add_patch(rect)
        if ps >= 12:  # only draw index text if patches are large enough to read
            ax0.text(meta.center_x, meta.center_y, str(meta.patch_index),
                      color="black", fontsize=6, ha="center", va="center")
    ax0.set_title(f"Voxel grid with patch boundaries\n"
                  f"({n_rows}x{n_cols} patches, size={ps}, top-{top_k_highlight} highlighted in gold)")
    ax0.axis("off")

    ax1 = axes[1]
    density_grid = np.zeros((n_rows, n_cols))
    for meta, s in zip(metadata, stats):
        density_grid[meta.row_index, meta.col_index] = s.event_density
    im = ax1.imshow(density_grid, cmap="inferno")
    ax1.set_title("Patch-level event density heatmap")
    fig.colorbar(im, ax=ax1, fraction=0.04)
    ax1.set_xlabel("patch column")
    ax1.set_ylabel("patch row")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
