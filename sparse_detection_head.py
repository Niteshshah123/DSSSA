"""
================================================================================
Sparse Anchor-Free Per-Token Detection Head (M9)
================================================================================
Applies a small MLP DIRECTLY to EDPS's surviving K tokens -- no
re-densification into a spatial grid anywhere in this module. Each token
predicts, relative to its OWN patch center (FCOS-style):

    (l, t, r, b)   distances from patch center to box edges
    class_logits   (num_classes,)
    objectness     scalar (is there an object at all)
    quality        scalar (localization quality / IoU-aware confidence)

INTERFACE INDEPENDENCE FROM THE BACKBONE (per requirement): `SparseDetectionHead`
only ever sees a (B, K, embedding_dim) tensor. It has no reference to
`TransformerEncoder`, its config, or anything Transformer-specific -- swapping
the backbone for a future SpikeFormer requires zero changes here, as long as
the replacement also emits (B, K, embedding_dim).

Two-stage design, deliberately kept separate:
  1. SparseDetectionHead.forward(tokens) -> raw tensor predictions. Pure
     tensor-in/tensor-out, trainable, backbone-agnostic.
  2. decode_predictions(...) -> human-readable structured detections,
     combining the raw predictions with M5's patch metadata (pixel centers)
     to produce absolute-coordinate boxes and preserve the token-to-patch
     mapping for visualization. This stays OUTSIDE the nn.Module so the
     network itself has no dependency on metadata bookkeeping.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from detection_head_config import DetectionHeadConfig
from patch_metadata import PatchMetadata


class SparseDetectionHead(nn.Module):
    def __init__(self, config: DetectionHeadConfig):
        super().__init__()
        self.config = config
        out_dim = 4 + config.num_classes + 1 + 1  # (l,t,r,b) + classes + objectness + quality

        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        tokens: (B, K, embedding_dim) -- from ANY backbone producing this shape.
        Returns raw (un-decoded) predictions:
          {"ltrb_raw": (B,K,4), "class_logits": (B,K,num_classes),
           "objectness_logit": (B,K,1), "quality_logit": (B,K,1)}
        """
        out = self.mlp(tokens)
        ltrb_raw, class_logits, objectness_logit, quality_logit = torch.split(
            out, [4, self.config.num_classes, 1, 1], dim=-1
        )
        return {
            "ltrb_raw": ltrb_raw,
            "class_logits": class_logits,
            "objectness_logit": objectness_logit,
            "quality_logit": quality_logit,
        }


@dataclass
class TokenDetection:
    """One decoded detection, with full token-to-patch provenance preserved."""
    patch_index: int
    row_index: int
    col_index: int
    center_x: float
    center_y: float
    box_x0: float
    box_y0: float
    box_x1: float
    box_y1: float
    class_probs: List[float]
    predicted_class: int
    objectness: float
    quality: float
    combined_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def decode_predictions(
    raw: Dict[str, torch.Tensor],
    patch_metadata: List[PatchMetadata],
    config: DetectionHeadConfig,
    attention_mask: Optional[torch.Tensor] = None,
    batch_index: int = 0,
) -> List[TokenDetection]:
    """
    Decodes ONE sample's raw predictions (indexed by batch_index) into
    absolute-coordinate detections, using M5's patch metadata for centers.

    raw: dict of (B, K, ...) tensors, as returned by SparseDetectionHead.forward
    patch_metadata: list of length equal to the number of VALID (non-padded)
      tokens for THIS sample -- e.g. EDPSOutput.selected_metadata -- NOT
      necessarily the batch's padded width K, since other samples in the
      batch may have more retained tokens than this one.
    attention_mask: optional (B, K) bool, True = real token. If provided,
      padded positions are located from the mask and never decoded; if
      None, every position in this sample is assumed valid.
    """
    ltrb_full = F.softplus(raw["ltrb_raw"][batch_index]) * config.distance_scale  # (K, 4)
    class_probs_full = F.softmax(raw["class_logits"][batch_index], dim=-1)        # (K, num_classes)
    objectness_full = torch.sigmoid(raw["objectness_logit"][batch_index]).squeeze(-1)  # (K,)
    quality_full = torch.sigmoid(raw["quality_logit"][batch_index]).squeeze(-1)        # (K,)

    k_padded = ltrb_full.shape[0]

    # Locate the VALID (non-padded) token positions within this sample's
    # padded slice. Built from the mask directly rather than assumed
    # contiguous, so this is correct regardless of where padding sits.
    if attention_mask is not None:
        valid_positions = [i for i in range(k_padded) if bool(attention_mask[batch_index, i])]
    else:
        valid_positions = list(range(k_padded))

    if len(patch_metadata) != len(valid_positions):
        raise ValueError(
            f"patch_metadata has {len(patch_metadata)} entries but there are "
            f"{len(valid_positions)} VALID (non-padded) tokens for this sample "
            f"(batch_index={batch_index}) -- they must correspond 1:1 in the "
            f"same order. (Note: the padded batch width is {k_padded}, which is "
            f"expected to differ from patch_metadata's length whenever other "
            f"samples in the batch retained more tokens than this one.)"
        )

    detections = []
    for meta_idx, token_idx in enumerate(valid_positions):
        meta = patch_metadata[meta_idx]
        l, t, r, b = ltrb_full[token_idx].tolist()
        cx, cy = meta.center_x, meta.center_y

        cp = class_probs_full[token_idx]
        predicted_class = int(torch.argmax(cp).item())
        max_class_prob = float(cp[predicted_class].item())
        obj = float(objectness_full[token_idx].item())
        qual = float(quality_full[token_idx].item())

        detections.append(TokenDetection(
            patch_index=meta.patch_index, row_index=meta.row_index, col_index=meta.col_index,
            center_x=cx, center_y=cy,
            box_x0=cx - l, box_y0=cy - t, box_x1=cx + r, box_y1=cy + b,
            class_probs=cp.tolist(), predicted_class=predicted_class,
            objectness=obj, quality=qual,
            combined_score=obj * qual * max_class_prob,
        ))

    return detections


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
