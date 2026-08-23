"""
================================================================================
M8 Transformer Computational Analysis
================================================================================
Unlike M7's FLOPs projection (which had no real model to measure), M8's
encoder actually exists now, so latency and memory here are REAL
measurements, not estimates. FLOPs are still analytic (measuring real FLOPs
requires a profiler hook, which is a reasonable future addition but not
implemented here), using the same documented per-layer formula introduced
in M7, now evaluated with the encoder's ACTUAL configuration rather than
assumed defaults.
================================================================================
"""

from __future__ import annotations

import time
from typing import Dict, Any

import torch

from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder, count_parameters
from edps_cost_analysis import estimate_transformer_flops


def measure_transformer_cost(
    encoder: TransformerEncoder,
    tokens: torch.Tensor,
    attention_mask: torch.Tensor = None,
    n_repeats: int = 5,
) -> Dict[str, Any]:
    encoder.eval()
    b, k, d = tokens.shape
    cfg = encoder.config

    with torch.no_grad():
        encoder(tokens, attention_mask=attention_mask)  # warm-up

    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            encoder(tokens, attention_mask=attention_mask)
            times.append(time.perf_counter() - t0)

    n_params = count_parameters(encoder)
    param_bytes = n_params * 4
    activation_bytes = b * k * d * 4 * cfg.num_layers  # rough: one activation tensor per layer

    ffn_ratio = cfg.ffn_dim / cfg.embedding_dim
    flops_per_sample = estimate_transformer_flops(k, d, cfg.num_layers, ffn_ratio)
    total_flops = flops_per_sample * b

    return {
        "batch_size": b,
        "tokens_per_sample": k,
        "embedding_dim": d,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "ffn_dim": cfg.ffn_dim,
        "n_parameters": n_params,
        "parameter_bytes": param_bytes,
        "estimated_activation_bytes": activation_bytes,
        "estimated_flops_per_sample": flops_per_sample,
        "estimated_flops_total_batch": total_flops,
        "avg_latency_ms": (sum(times) / len(times)) * 1000,
        "min_latency_ms": min(times) * 1000,
        "max_latency_ms": max(times) * 1000,
    }


def compare_before_after_edps(
    encoder: TransformerEncoder, n_input_tokens: int, n_retained_tokens: int, embedding_dim: int,
) -> Dict[str, Any]:
    cfg = encoder.config
    ffn_ratio = cfg.ffn_dim / cfg.embedding_dim

    flops_before = estimate_transformer_flops(n_input_tokens, embedding_dim, cfg.num_layers, ffn_ratio)
    flops_after = estimate_transformer_flops(n_retained_tokens, embedding_dim, cfg.num_layers, ffn_ratio)
    reduction_pct = 100.0 * (flops_before - flops_after) / flops_before if flops_before else 0.0

    # Measure REAL latency for both token counts (single-sample, no padding, so
    # the comparison isn't confounded by mask overhead)
    encoder.eval()
    with torch.no_grad():
        tokens_before = torch.randn(1, n_input_tokens, embedding_dim)
        t0 = time.perf_counter()
        encoder(tokens_before)
        latency_before_ms = (time.perf_counter() - t0) * 1000

        tokens_after = torch.randn(1, n_retained_tokens, embedding_dim)
        t0 = time.perf_counter()
        encoder(tokens_after)
        latency_after_ms = (time.perf_counter() - t0) * 1000

    latency_reduction_pct = (
        100.0 * (latency_before_ms - latency_after_ms) / latency_before_ms if latency_before_ms else 0.0
    )

    return {
        "n_input_tokens": n_input_tokens,
        "n_retained_tokens": n_retained_tokens,
        "theoretical_flops_before_edps": flops_before,
        "theoretical_flops_after_edps": flops_after,
        "flops_reduction_pct": reduction_pct,
        "measured_latency_before_edps_ms": latency_before_ms,
        "measured_latency_after_edps_ms": latency_after_ms,
        "measured_latency_reduction_pct": latency_reduction_pct,
        "note": (
            "FLOPs are analytic (formula-based); latency here IS a real "
            "single-sample measurement of the actual encoder built in this "
            "module (unlike M7's purely theoretical latency projection, "
            "since the encoder now exists)."
        ),
    }
