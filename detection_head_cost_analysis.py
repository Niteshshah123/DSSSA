"""
================================================================================
M9 Detection Head Computational Analysis
================================================================================
Real measurements (the head exists, unlike M7's pre-Transformer projection).
FLOPs are analytic for the MLP (simple to compute exactly, unlike attention).
================================================================================
"""

from __future__ import annotations

import time
from typing import Dict, Any

import torch

from sparse_detection_head import SparseDetectionHead, count_parameters


def estimate_head_flops(n_tokens: int, embedding_dim: int, hidden_dim: int, num_classes: int) -> int:
    out_dim = 4 + num_classes + 1 + 1
    # 3 linear layers: (D->H), (H->H), (H->out_dim); multiply-add counted as 2 FLOPs
    flops = n_tokens * (
        embedding_dim * hidden_dim * 2
        + hidden_dim * hidden_dim * 2
        + hidden_dim * out_dim * 2
    )
    return flops


def measure_detection_head_cost(
    head: SparseDetectionHead, tokens: torch.Tensor, n_repeats: int = 5,
) -> Dict[str, Any]:
    head.eval()
    b, k, d = tokens.shape
    cfg = head.config

    with torch.no_grad():
        head(tokens)  # warm-up

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            head(tokens)
            times.append(time.perf_counter() - t0)

    n_params = count_parameters(head)
    param_bytes = n_params * 4
    out_dim = 4 + cfg.num_classes + 1 + 1
    output_bytes = b * k * out_dim * 4

    flops_per_sample = estimate_head_flops(k, d, cfg.hidden_dim, cfg.num_classes)
    total_flops = flops_per_sample * b

    return {
        "batch_size": b,
        "tokens_per_sample": k,
        "embedding_dim": d,
        "hidden_dim": cfg.hidden_dim,
        "num_classes": cfg.num_classes,
        "n_parameters": n_params,
        "parameter_bytes": param_bytes,
        "output_bytes": output_bytes,
        "estimated_flops_per_sample": flops_per_sample,
        "estimated_flops_total_batch": total_flops,
        "avg_latency_ms": (sum(times) / len(times)) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
    }
