"""
================================================================================
EDPS Visualization
================================================================================
Four panels:
  1. Importance score heatmap over the patch grid (uses row/col from metadata,
     independent of pixel-level voxel imagery).
  2. Binary selection mask (green=retained, red=removed) over the same grid.
  3. Importance score histogram, with the gate threshold marked.
  4. Token reduction statistics as a text summary panel.
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch

from patch_metadata import PatchMetadata


def visualize_edps(
    importance_scores: torch.Tensor,
    binary_mask: torch.Tensor,
    patch_metadata: List[PatchMetadata],
    n_rows: int,
    n_cols: int,
    gate_threshold: float,
    token_reduction_stats: Dict[str, Any],
    out_path: str = "edps_visualization.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = importance_scores.detach().numpy() if torch.is_tensor(importance_scores) else np.asarray(importance_scores)
    mask = binary_mask.detach().numpy() if torch.is_tensor(binary_mask) else np.asarray(binary_mask)

    score_grid = np.zeros((n_rows, n_cols))
    mask_grid = np.zeros((n_rows, n_cols))
    for m, s, k in zip(patch_metadata, scores, mask):
        score_grid[m.row_index, m.col_index] = s
        mask_grid[m.row_index, m.col_index] = 1.0 if k else 0.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    im0 = axes[0].imshow(score_grid, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("Importance score heatmap")
    fig.colorbar(im0, ax=axes[0], fraction=0.04)

    im1 = axes[1].imshow(mask_grid, cmap="RdYlGn", vmin=0, vmax=1)
    axes[1].set_title(f"Selection mask\n(green=retained, red=removed)")

    axes[2].hist(scores, bins=40, color="slateblue")
    axes[2].axvline(gate_threshold, color="red", linestyle="--", label=f"threshold={gate_threshold}")
    axes[2].set_title("Importance score histogram")
    axes[2].set_xlabel("score")
    axes[2].legend()

    axes[3].axis("off")
    lines = [
        f"Input tokens: {token_reduction_stats['n_input']}",
        f"Retained: {token_reduction_stats['n_retained']}",
        f"Removed: {token_reduction_stats['n_removed']}",
        f"% retained: {token_reduction_stats['pct_retained']:.2f}%",
        f"% removed: {token_reduction_stats['pct_removed']:.2f}%",
        f"Safety clamp triggered: {token_reduction_stats['clamp_triggered']}",
        f"Clamp direction: {token_reduction_stats['clamp_direction']}",
    ]
    axes[3].text(0.05, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    axes[3].set_title("Token reduction statistics")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
