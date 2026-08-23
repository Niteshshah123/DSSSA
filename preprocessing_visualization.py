"""
================================================================================
Preprocessing Before/After Visualization (M4, item 6)
================================================================================
Renders one row per stage (raw -> after BAF -> after hot-pixel filter ->
after isolated-event removal -> final), each panel showing the surviving
events (red=ON, blue=OFF) with ground-truth boxes overlaid, plus a header
reporting how many events remain and the cumulative % removed.
================================================================================
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def visualize_preprocessing_stages(
    stage_arrays: List[dict],   # list of {"name": str, "x": arr, "y": arr, "p": arr}
    height: int,
    width: int,
    boxes: Optional[np.ndarray] = None,
    out_path: str = "preprocessing_stages.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(stage_arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    n0 = len(stage_arrays[0]["x"]) if n > 0 else 0

    for ax, stage in zip(axes, stage_arrays):
        frame = np.full((height, width, 3), 255, dtype=np.uint8)
        x, y, p = stage["x"], stage["y"], stage["p"]
        if len(x) > 0:
            frame[y[p > 0], x[p > 0]] = [255, 0, 0]
            frame[y[p <= 0], x[p <= 0]] = [0, 0, 255]
        ax.imshow(frame)
        if boxes is not None:
            for b in boxes:
                rect = patches.Rectangle((b["x"], b["y"]), b["w"], b["h"],
                                          linewidth=1.2, edgecolor="lime", facecolor="none")
                ax.add_patch(rect)
        n_now = len(x)
        pct_removed_cumulative = (100.0 * (n0 - n_now) / n0) if n0 else 0.0
        ax.set_title(f"{stage['name']}\n{n_now} events "
                     f"(-{pct_removed_cumulative:.1f}% cum.)", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
