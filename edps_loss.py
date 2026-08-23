"""
================================================================================
EDPS Sparsity Loss (hinge band)
================================================================================
loss = relu(min_ratio - r) + relu(r - max_ratio),  r = mean(soft_gate)

Zero penalty as long as the batch's mean soft keep-rate stays inside
[min_ratio, max_ratio] -- this is what lets K float with scene complexity
rather than being pulled toward one fixed target ratio (unlike, e.g.,
DynamicViT's precise target-ratio loss). Only penalizes if the network
drifts to a degenerate extreme (keeping everything or nothing).

Uses the CONTINUOUS soft gate (pre-threshold), never the hard binary mask,
so it remains differentiable w.r.t. the scoring network's parameters.
================================================================================
"""

from __future__ import annotations

import torch


def sparsity_loss(soft_gates: torch.Tensor, min_ratio: float, max_ratio: float) -> torch.Tensor:
    """soft_gates: (N,) or (B, N), values in (0, 1). Returns a scalar loss."""
    r = soft_gates.mean()
    below = torch.clamp(min_ratio - r, min=0.0)
    above = torch.clamp(r - max_ratio, min=0.0)
    return below + above
