"""
================================================================================
Transformer Encoder (M8)
================================================================================
Pre-LayerNorm residual blocks, no CLS token, no internal positional
embedding (M6 already injected positional + temporal signal). Supports
variable-length token sequences via a boolean attention_mask, and can
optionally return every layer's attention weights for later visualization.

ATTENTION MASK CONVENTION (public API): attention_mask is (B, K), True =
REAL token, False = padding. This is the natural "keep mask" convention,
matching EDPS's own binary_mask semantics. Internally, this is inverted
before being passed to nn.MultiheadAttention's key_padding_mask, which uses
the OPPOSITE convention (True = ignore this position) -- the inversion
happens once, here, so callers never have to think about it.

INTERFACE (exactly as requested, for future SpikeFormer swap):
    TransformerEncoder.forward(tokens, attention_mask=None, return_attention=False)
        -> tokens                                   (if return_attention=False)
        -> (tokens, [attn_layer_0, ..., attn_layer_{L-1}])  (if True)
    Each attn_layer_i has shape (B, num_heads, K, K).
================================================================================
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn

from transformer_config import TransformerEncoderConfig


class PreLNEncoderBlock(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim, num_heads=config.num_heads,
            dropout=config.dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embedding_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor], return_attention: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # --- Pre-LN attention sub-block ---
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,  # keep per-head weights: (B, num_heads, K, K)
        )
        x = x + attn_out

        # --- Pre-LN feed-forward sub-block ---
        x = x + self.ffn(self.norm2(x))

        return x, (attn_weights if return_attention else None)


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerEncoderConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([PreLNEncoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        tokens: (B, K, embedding_dim)
        attention_mask: (B, K) bool, True = real token, False = padding. If
          None, every token is treated as real (no masking applied).
        """
        key_padding_mask = (~attention_mask) if attention_mask is not None else None

        x = tokens
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x, key_padding_mask=key_padding_mask, return_attention=return_attention)
            if return_attention:
                attn_maps.append(attn)

        x = self.final_norm(x)

        if return_attention:
            return x, attn_maps
        return x


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
