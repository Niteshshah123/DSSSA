"""
================================================================================
Embedding Visualization (M6, item 6)
================================================================================
Four panels:
  1. Feature value distribution (histogram across all embedding dimensions)
  2. PCA projection of patches to 2D, colored by total activity (if provided)
  3. t-SNE projection to 2D (optional -- can be slow/degenerate for very
     small N, so it's skipped gracefully rather than erroring)
  4. Cosine similarity heatmap between all patches
================================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def visualize_embeddings(
    embeddings: torch.Tensor,
    out_path: str = "embedding_visualization.png",
    activity: Optional[np.ndarray] = None,
    include_tsne: bool = True,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    emb = embeddings.detach().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)
    n, d = emb.shape

    n_panels = 4 if include_tsne else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    # 1. Feature distribution
    axes[0].hist(emb.flatten(), bins=60, color="steelblue")
    axes[0].set_title(f"Embedding value distribution\n(dim={d}, N={n} patches)")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("count")

    # 2. PCA projection
    color = activity if activity is not None else np.zeros(n)
    if n >= 2:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(emb)
        sc = axes[1].scatter(proj[:, 0], proj[:, 1], c=color, cmap="viridis", s=15)
        axes[1].set_title(f"PCA projection\n(explained var: {pca.explained_variance_ratio_.sum():.2f})")
        fig.colorbar(sc, ax=axes[1], fraction=0.04, label="activity" if activity is not None else None)
    else:
        axes[1].text(0.5, 0.5, "Not enough patches for PCA", ha="center", va="center")
        axes[1].axis("off")

    # 3. t-SNE (optional, skipped gracefully if N too small)
    panel_idx = 2
    if include_tsne:
        if n >= 10:
            try:
                from sklearn.manifold import TSNE
                perplexity = min(30, max(5, n // 4))
                tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0)
                proj_tsne = tsne.fit_transform(emb)
                sc2 = axes[2].scatter(proj_tsne[:, 0], proj_tsne[:, 1], c=color, cmap="viridis", s=15)
                axes[2].set_title("t-SNE projection")
                fig.colorbar(sc2, ax=axes[2], fraction=0.04)
            except Exception as e:
                axes[2].text(0.5, 0.5, f"t-SNE failed:\n{e}", ha="center", va="center", fontsize=8)
                axes[2].axis("off")
        else:
            axes[2].text(0.5, 0.5, f"Skipped: need >=10 patches for t-SNE\n(got {n})",
                          ha="center", va="center", fontsize=9)
            axes[2].axis("off")
        panel_idx = 3

    # 4. Cosine similarity heatmap
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    im = axes[panel_idx].imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    axes[panel_idx].set_title("Cosine similarity between patches")
    fig.colorbar(im, ax=axes[panel_idx], fraction=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
