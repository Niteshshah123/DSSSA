"""
================================================================================
Detection Visualization (M9)
================================================================================
Four panels:
  1. Retained patch grid (which tokens survived EDPS -- context for the rest)
  2. Predicted boxes overlaid on a blank sensor-sized canvas, colored by
     predicted class, labeled with confidence (combined_score)
  3. Token-to-detection mapping: draws a line from each source patch center
     to its predicted box center, making the provenance visually explicit
  4. Confidence score distribution (histogram of combined_score across all
     decoded detections in this sample)
================================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import matplotlib.patches as mpatches

from sparse_detection_head import TokenDetection
from patch_metadata import PatchMetadata

CLASS_COLORS = ["lime", "orange", "cyan", "magenta", "yellow", "red"]


def visualize_detections(
    detections: List[TokenDetection],
    all_patch_metadata: List[PatchMetadata],  # FULL grid (pre-EDPS), for the retained-patch panel
    selected_metadata: List[PatchMetadata],   # SELECTED (post-EDPS) subset actually fed to the head
    n_rows: int,
    n_cols: int,
    sensor_height: int,
    sensor_width: int,
    score_threshold: float = 0.0,
    out_path: str = "detection_visualization.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # --- Panel 1: retained patch grid ---
    retained_grid = np.zeros((n_rows, n_cols))
    selected_indices = {m.patch_index for m in selected_metadata}
    for m in all_patch_metadata:
        retained_grid[m.row_index, m.col_index] = 1.0 if m.patch_index in selected_indices else 0.0
    axes[0].imshow(retained_grid, cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_title(f"Retained patches (EDPS)\n{len(selected_metadata)}/{len(all_patch_metadata)} kept")

    # --- Panel 2: predicted boxes on sensor-sized canvas ---
    canvas = np.full((sensor_height, sensor_width, 3), 255, dtype=np.uint8)
    axes[1].imshow(canvas)
    shown = [d for d in detections if d.combined_score >= score_threshold]
    for d in shown:
        color = CLASS_COLORS[d.predicted_class % len(CLASS_COLORS)]
        rect = mpatches.Rectangle(
            (d.box_x0, d.box_y0), d.box_x1 - d.box_x0, d.box_y1 - d.box_y0,
            linewidth=1.2, edgecolor=color, facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].text(d.box_x0, max(d.box_y0 - 2, 0),
                     f"cls{d.predicted_class}:{d.combined_score:.2f}",
                     color=color, fontsize=6)
    axes[1].set_title(f"Predicted boxes (score >= {score_threshold})\n{len(shown)}/{len(detections)} shown")
    axes[1].set_xlim(0, sensor_width)
    axes[1].set_ylim(sensor_height, 0)

    # --- Panel 3: token-to-detection mapping ---
    axes[2].imshow(canvas)
    for d in shown:
        axes[2].plot([d.center_x, (d.box_x0 + d.box_x1) / 2],
                     [d.center_y, (d.box_y0 + d.box_y1) / 2],
                     color="steelblue", linewidth=0.6, alpha=0.6)
        axes[2].scatter([d.center_x], [d.center_y], color="black", s=8, zorder=3)
        color = CLASS_COLORS[d.predicted_class % len(CLASS_COLORS)]
        rect = mpatches.Rectangle(
            (d.box_x0, d.box_y0), d.box_x1 - d.box_x0, d.box_y1 - d.box_y0,
            linewidth=1.0, edgecolor=color, facecolor="none",
        )
        axes[2].add_patch(rect)
    axes[2].set_title("Token center -> predicted box mapping")
    axes[2].set_xlim(0, sensor_width)
    axes[2].set_ylim(sensor_height, 0)

    # --- Panel 4: confidence score histogram ---
    scores = [d.combined_score for d in detections]
    axes[3].hist(scores, bins=30, color="slateblue")
    axes[3].axvline(score_threshold, color="red", linestyle="--", label=f"threshold={score_threshold}")
    axes[3].set_title("Combined confidence score distribution")
    axes[3].set_xlabel("combined_score")
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
