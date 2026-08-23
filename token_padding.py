"""
================================================================================
Token Padding Utility (M8 <-> M7 integration)
================================================================================
EDPS produces a different K per sample. To batch them for the Transformer,
pad every sample up to the batch's max K with zero vectors, and build the
boolean attention_mask (True = real token) that tells the encoder to
ignore the padding.
================================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def pad_token_sequences(embeddings_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    embeddings_list: list of (K_i, D) tensors, K_i possibly different per sample.
    Returns:
      padded: (B, K_max, D), zero-padded
      attention_mask: (B, K_max) bool, True = real token, False = padding
    """
    if len(embeddings_list) == 0:
        raise ValueError("embeddings_list is empty")

    d = embeddings_list[0].shape[1]
    k_max = max(e.shape[0] for e in embeddings_list)
    b = len(embeddings_list)

    padded = torch.zeros(b, k_max, d, dtype=embeddings_list[0].dtype)
    attention_mask = torch.zeros(b, k_max, dtype=torch.bool)

    for i, emb in enumerate(embeddings_list):
        k = emb.shape[0]
        padded[i, :k] = emb
        attention_mask[i, :k] = True

    return padded, attention_mask
