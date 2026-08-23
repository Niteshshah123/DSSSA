"""
================================================================================
M6 Computational Analysis (item 7)
================================================================================
Reports parameter count, FLOPs estimate, measured latency, and memory
consumption for a given PatchEmbeddingModule configuration. FLOPs are
estimated analytically per projection type (a real forward pass is also
timed for latency, since analytic FLOPs and wall-clock time capture
different things).
================================================================================
"""

from __future__ import annotations

import time
from typing import Dict, Any

import torch

from patch_embedding_module import PatchEmbeddingModule, count_parameters


def estimate_embedding_flops(
    n_patches: int, in_channels: int, patch_size: int, embedding_dim: int,
    projection_type: str, cnn_hidden_channels: int = 32,
) -> int:
    """Analytic FLOPs estimate (multiply-adds counted as 2 FLOPs each), projection only."""
    if projection_type == "linear":
        in_features = in_channels * patch_size * patch_size
        return n_patches * in_features * embedding_dim * 2

    if projection_type == "cnn":
        # two 3x3 conv2d layers (in->hidden, hidden->hidden) + final linear
        conv1 = n_patches * (patch_size * patch_size) * in_channels * cnn_hidden_channels * 9 * 2
        conv2 = n_patches * (patch_size * patch_size) * cnn_hidden_channels * cnn_hidden_channels * 9 * 2
        linear = n_patches * cnn_hidden_channels * embedding_dim * 2
        return conv1 + conv2 + linear

    if projection_type == "conv3d":
        hidden = cnn_hidden_channels // 2 if cnn_hidden_channels >= 2 else cnn_hidden_channels  # matches default hidden=16
        conv3d = n_patches * in_channels * (patch_size * patch_size) * hidden * 27 * 2  # 3x3x3 kernel
        linear = n_patches * hidden * embedding_dim * 2
        return conv3d + linear

    raise ValueError(f"Unknown projection_type {projection_type!r}")


def measure_embedding_module_cost(
    module: PatchEmbeddingModule, patches: torch.Tensor, metadata, n_repeats: int = 5,
) -> Dict[str, Any]:
    """Measures real forward-pass latency and reports parameter/memory stats."""
    n_patches, c, ps, _ = patches.shape

    # warm-up
    with torch.no_grad():
        module(patches, metadata)

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            module(patches, metadata)
            times.append(time.perf_counter() - t0)

    n_params = count_parameters(module)
    param_bytes = n_params * 4  # float32
    output_bytes = n_patches * module.config.embedding_dim * 4

    flops = estimate_embedding_flops(
        n_patches, c, ps, module.config.embedding_dim,
        module.config.projection_type, module.config.cnn_hidden_channels,
    )

    return {
        "n_patches": n_patches,
        "embedding_dim": module.config.embedding_dim,
        "projection_type": module.config.projection_type,
        "n_parameters": n_params,
        "parameter_bytes": param_bytes,
        "output_bytes": output_bytes,
        "estimated_flops": flops,
        "avg_latency_ms": (sum(times) / len(times)) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
    }
