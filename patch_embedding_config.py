"""
================================================================================
M6 Patch Embedding Configuration
================================================================================
Nothing hardcoded: embedding dimension, projection strategy, positional
encoding strategy, and temporal encoding strategy/toggle are all exposed
here. Patch size and channel count are NEVER assumed constant -- every
downstream class in this module reads them from the actual input tensor
shape at runtime.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchEmbeddingConfig:
    embedding_dim: int = 256

    # Item 2: projection strategy -- must match a key registered in
    # patch_projections.PROJECTION_REGISTRY
    projection_type: str = "linear"          # "linear" | "cnn" | "conv3d"
    cnn_hidden_channels: int = 32            # used by "cnn" and "conv3d" projections

    # Item 3: positional encoding
    positional_encoding_type: str = "learnable"   # "learnable" | "sinusoidal"

    # Item 4: temporal encoding (optional)
    use_temporal_encoding: bool = True
    temporal_encoding_type: str = "learnable"     # "learnable" | "sinusoidal"

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.projection_type not in ("linear", "cnn", "conv3d"):
            raise ValueError(f"projection_type must be 'linear'/'cnn'/'conv3d', got {self.projection_type!r}")
        if self.positional_encoding_type not in ("learnable", "sinusoidal"):
            raise ValueError(f"positional_encoding_type must be 'learnable'/'sinusoidal', got {self.positional_encoding_type!r}")
        if self.temporal_encoding_type not in ("learnable", "sinusoidal"):
            raise ValueError(f"temporal_encoding_type must be 'learnable'/'sinusoidal', got {self.temporal_encoding_type!r}")
