"""
================================================================================
Attention Map Visualization (M8)
================================================================================
Renders one panel per encoder layer (averaged over heads, for readability),
showing the (K, K) attention matrix for a single sample. Since M8 has no
detection head yet, this is the earliest point real attention behavior can
be inspected.
================================================================================
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def visualize_attention_maps(
    attention_maps: List[torch.Tensor],
    sample_index: int = 0,
    n_valid_tokens: int = None,
    out_path: str = "attention_maps.png",
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(attention_maps)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4.5))
    if n_layers == 1:
        axes = [axes]

    for i, attn in enumerate(attention_maps):
        # attn: (B, num_heads, K, K)
        a = attn[sample_index].detach().numpy() if torch.is_tensor(attn) else np.asarray(attn[sample_index])
        avg_over_heads = a.mean(axis=0)  # (K, K)
        if n_valid_tokens is not None:
            avg_over_heads = avg_over_heads[:n_valid_tokens, :n_valid_tokens]

        im = axes[i].imshow(avg_over_heads, cmap="viridis")
        axes[i].set_title(f"Layer {i} (mean over heads)")
        axes[i].set_xlabel("key token")
        axes[i].set_ylabel("query token")
        fig.colorbar(im, ax=axes[i], fraction=0.046)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
