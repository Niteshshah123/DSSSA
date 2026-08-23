"""
================================================================================
Batch Visualization (M2, item 8)
================================================================================
Renders every sample in a batch (as produced by dataloader_utils.collate_events,
i.e. a list of per-sample dicts) as one accumulated event frame + box overlay,
arranged in a grid, and saves it as a single PNG. Reuses the same rendering
convention as EventParser.visualize_events (red = ON/polarity 1, blue =
OFF/polarity 0) for visual consistency across the whole project.
================================================================================
"""

from __future__ import annotations

import math
from typing import List, Dict, Any

import numpy as np


def visualize_batch(
    batch: List[Dict[str, Any]],
    out_path: str = "batch_visualization.png",
    max_samples: int = 16,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    samples = batch[:max_samples]
    n = len(samples)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for i, sample in enumerate(samples):
        ax = axes[i]
        h, w = sample["sensor_height"], sample["sensor_width"]
        frame = np.full((h, w, 3), 255, dtype=np.uint8)

        x = sample["x"].numpy()
        y = sample["y"].numpy()
        p = sample["p"].numpy()
        frame[y[p == 1], x[p == 1]] = [255, 0, 0]
        frame[y[p == 0], x[p == 0]] = [0, 0, 255]

        ax.imshow(frame)
        for b in sample["boxes"]:
            rect = patches.Rectangle(
                (b["x"], b["y"]), b["w"], b["h"],
                linewidth=1.2, edgecolor="lime", facecolor="none",
            )
            ax.add_patch(rect)

        ax.set_title(
            f"{sample['recording_stem']} win#{sample['window_index']}\n"
            f"{len(sample['t'])} events, {len(sample['boxes'])} boxes",
            fontsize=8,
        )
        ax.axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
