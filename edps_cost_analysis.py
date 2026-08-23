"""
================================================================================
EDPS Computational Analysis
================================================================================
Estimates the Transformer FLOPs reduction EDPS is projected to enable, since
the actual Transformer (M8) doesn't exist yet. These are THEORETICAL
PROJECTIONS based on a standard, documented per-layer Transformer FLOPs
formula -- not measurements of a real model. They should be re-validated
empirically once M8 exists; this module says so explicitly rather than
implying otherwise.

FLOPS FORMULA (standard approximation for one Transformer encoder layer,
sequence length N, embedding dim D, feed-forward ratio r, counting each
multiply-add as 2 FLOPs):
    QKV + output projections:  4 * N * D^2 * 2
    Attention matmuls (QK^T and AV):  2 * (N^2 * D) * 2
    Feed-forward (two linear layers, hidden = r*D):  2 * (N * D * r*D) * 2
Total per layer = 8*N*D^2 + 4*N^2*D + 4*r*N*D^2
Multiply by n_layers for the full stack.

The N^2 term is what EDPS specifically targets: it scales quadratically
with token count, so shrinking N -> K (K << N) gives the largest relative
savings on exactly that term, growing more favorable as N increases.

LATENCY REDUCTION ESTIMATE: since the N^2 attention term dominates for
large N, a simple projected latency reduction is 1 - (K/N)^2 applied to
that term's share of total FLOPs -- reported as a rough estimate, not a
measured wall-clock number (measuring that requires M8 to exist).
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any


def estimate_transformer_flops(n_tokens: int, embedding_dim: int, n_layers: int = 4, ffn_ratio: int = 4) -> int:
    d = embedding_dim
    per_layer = (
        8 * n_tokens * d * d
        + 4 * (n_tokens ** 2) * d
        + 4 * ffn_ratio * n_tokens * d * d
    )
    return per_layer * n_layers


def compute_edps_flops_comparison(
    n_input_tokens: int, n_retained_tokens: int, embedding_dim: int,
    n_layers: int = 4, ffn_ratio: int = 4,
) -> Dict[str, Any]:
    flops_before = estimate_transformer_flops(n_input_tokens, embedding_dim, n_layers, ffn_ratio)
    flops_after = estimate_transformer_flops(n_retained_tokens, embedding_dim, n_layers, ffn_ratio)
    flops_reduction_pct = 100.0 * (flops_before - flops_after) / flops_before if flops_before else 0.0

    ratio_sq = (n_retained_tokens / n_input_tokens) ** 2 if n_input_tokens else 1.0
    projected_latency_reduction_pct = 100.0 * (1.0 - ratio_sq)

    return {
        "n_input_tokens": n_input_tokens,
        "n_retained_tokens": n_retained_tokens,
        "token_reduction_pct": 100.0 * (n_input_tokens - n_retained_tokens) / n_input_tokens if n_input_tokens else 0.0,
        "estimated_transformer_flops_before": flops_before,
        "estimated_transformer_flops_after": flops_after,
        "flops_reduction_pct": flops_reduction_pct,
        "projected_latency_reduction_pct_theoretical": projected_latency_reduction_pct,
        "assumptions": {
            "n_layers": n_layers, "ffn_ratio": ffn_ratio, "embedding_dim": embedding_dim,
            "formula": "8*N*D^2 + 4*N^2*D + 4*ffn_ratio*N*D^2, per layer",
        },
        "caveat": (
            "These are analytic projections based on a standard Transformer FLOPs "
            "formula, NOT measurements of a real model -- M8 (Transformer backbone) "
            "does not exist yet. Re-validate empirically once it does, since real "
            "wall-clock speedup also depends on batching/padding strategy for "
            "variable per-sample K (see M7 design discussion)."
        ),
    }
