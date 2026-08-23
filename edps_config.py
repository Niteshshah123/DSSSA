"""
================================================================================
M7 EDPS Configuration
================================================================================
Per the approved design:
  - gate_threshold is configurable (refinement 1), not hardcoded, for later
    ablation.
  - clamp_min_ratio / clamp_max_ratio are a SAFETY MECHANISM ONLY. They do
    not implement top-K selection -- they only force a correction if the
    independent per-patch threshold decision produces a degenerate count
    (everything kept, or everything dropped). Default bounds are
    deliberately wide (0.05-0.95) so they rarely engage in normal operation;
    see edps_module.py docstring for exactly when they trigger.
  - sparsity_min_ratio / sparsity_max_ratio define the hinge BAND for the
    sparsity loss -- zero penalty inside the band, so K is free to float
    with scene complexity rather than being pulled toward one fixed ratio.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EDPSConfig:
    # ---- Feature construction ----
    local_feature_dim: int = 32      # dim of the per-patch projected local feature h_i

    # ---- Scoring network ----
    scoring_hidden_dim: int = 32
    use_neighbor_context: bool = True

    # ---- Gating ----
    gate_threshold: float = 0.5      # configurable per refinement 1; NOT hardcoded downstream
    selection_mode: str = "normal"   # "normal" | "baseline_no_pruning"

    # ---- Safety clamp (NOT the primary selection mechanism -- see module docstring) ----
    clamp_min_ratio: float = 0.05
    clamp_max_ratio: float = 0.95

    # ---- Sparsity loss (hinge band) ----
    sparsity_min_ratio: float = 0.2
    sparsity_max_ratio: float = 0.6
    lambda_sparsity: float = 0.1

    # ---- Diagnostics ----
    collapse_correlation_warn_threshold: float = 0.9  # |corr(score, density)| above this is flagged

    def __post_init__(self):
        if not (0.0 < self.gate_threshold < 1.0):
            raise ValueError(f"gate_threshold must be in (0,1), got {self.gate_threshold}")
        if self.selection_mode not in ("normal", "baseline_no_pruning"):
            raise ValueError(f"selection_mode must be 'normal' or 'baseline_no_pruning', got {self.selection_mode!r}")
        if not (0.0 <= self.clamp_min_ratio <= self.clamp_max_ratio <= 1.0):
            raise ValueError(
                f"Require 0 <= clamp_min_ratio <= clamp_max_ratio <= 1, "
                f"got {self.clamp_min_ratio}, {self.clamp_max_ratio}"
            )
        if not (0.0 <= self.sparsity_min_ratio <= self.sparsity_max_ratio <= 1.0):
            raise ValueError(
                f"Require 0 <= sparsity_min_ratio <= sparsity_max_ratio <= 1, "
                f"got {self.sparsity_min_ratio}, {self.sparsity_max_ratio}"
            )
        if self.lambda_sparsity < 0:
            raise ValueError(f"lambda_sparsity must be non-negative, got {self.lambda_sparsity}")
