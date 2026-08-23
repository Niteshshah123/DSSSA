"""
================================================================================
M8 Transformer Encoder Configuration
================================================================================
Fixed per your approved spec: dim=256, heads=8, layers=4, ffn=512, dropout=0.1,
Pre-LayerNorm, no CLS token, no internal positional embedding (M6 already
injected positional + temporal signal into every embedding). Still exposed
as a dataclass rather than hardcoded, so depth/width can be revisited in
later ablations without touching the encoder's code.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformerEncoderConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    norm_style: str = "pre"   # "pre" only is supported/approved; kept explicit rather than silently assumed

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.norm_style != "pre":
            raise ValueError(f"Only 'pre' (Pre-LayerNorm) is implemented/approved, got {self.norm_style!r}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
