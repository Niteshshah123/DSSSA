"""
================================================================================
M9 Sparse Detection Head Configuration
================================================================================
Every dimension/scale factor exposed, nothing hardcoded. num_classes
defaults to 2 (car, pedestrian -- confirmed via M0's dataset audit), but is
not assumed anywhere else in the code.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionHeadConfig:
    embedding_dim: int = 256          # must match the Transformer/backbone's output dim
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.1

    # FCOS-style (l, t, r, b) distance regression: raw network output is
    # passed through softplus (>=0) then multiplied by this factor to bring
    # it into pixel-scale before being combined with the patch's spatial_size.
    # Distances are predicted in UNITS OF PATCH SIZE (i.e. "how many patch
    # widths from my center is this edge"), so distance_scale is a
    # multiplier matching patch_size (16.0) -- keeping the raw regression target
    # roughly O(1) regardless of patch_size, which is friendlier to train.
    distance_scale: float = 16.0

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        if self.distance_scale <= 0:
            raise ValueError(f"distance_scale must be positive, got {self.distance_scale}")
