"""
================================================================================
Target Assignment (M10)
================================================================================
FCOS-style: a token is POSITIVE if its patch center falls inside a
ground-truth box. Assignment operates on the SELECTED (post-EDPS) tokens'
metadata only, since that's what the detection head actually predicts for.

IMPORTANT INTERACTION WITH EDPS (tracked explicitly, not silently ignored):
if EDPS drops every token whose center falls inside some GT box, that box
has ZERO positive candidates in this window and contributes NOTHING to the
detection loss -- a real, possible failure mode where pruning costs recall.
`assign_targets` returns not just per-token targets but also the count of
GT boxes with zero surviving positive tokens, so this can be logged as a
standing diagnostic throughout training (a "recall ceiling imposed by
EDPS" metric) rather than silently hidden inside an aggregate loss number.

Ambiguity rule: if a token's center falls inside multiple GT boxes, it's
assigned to the SMALLEST-area one (standard FCOS convention -- prefers the
more specific/local object over a larger one that happens to overlap).
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from patch_metadata import PatchMetadata


@dataclass
class AssignmentResult:
    is_positive: torch.Tensor          # (K,) bool
    target_class: torch.Tensor         # (K,) int64 (only meaningful where is_positive)
    target_ltrb: torch.Tensor          # (K,4) float32 (only meaningful where is_positive)
    target_box: torch.Tensor           # (K,4) float32 absolute (x0,y0,x1,y1), only meaningful where is_positive
    n_gt_boxes: int
    n_unmatched_gt_boxes: int          # GT boxes with zero surviving positive tokens


def assign_targets(
    selected_metadata: List[PatchMetadata], boxes: np.ndarray,
) -> AssignmentResult:
    """
    NOTE: target_ltrb is in REAL PIXEL UNITS (distances from patch center to
    box edges), matching the space the detection head's DECODED prediction
    (softplus(raw) * distance_scale) lives in -- not the network's raw
    pre-activation output. Losses are computed post-decode throughout
    (matching GIoU, which is unambiguously a pixel-space quantity), so no
    distance_scale division belongs here.
    """
    k = len(selected_metadata)
    is_positive = torch.zeros(k, dtype=torch.bool)
    target_class = torch.zeros(k, dtype=torch.long)
    target_ltrb = torch.zeros(k, 4, dtype=torch.float32)
    target_box = torch.zeros(k, 4, dtype=torch.float32)

    n_gt = len(boxes)
    matched_gt = set()

    if n_gt > 0 and k > 0:
        gt_areas = boxes["w"] * boxes["h"]
        for i, meta in enumerate(selected_metadata):
            cx, cy = meta.center_x, meta.center_y
            best_gt_idx = -1
            best_area = float("inf")
            for gi, b in enumerate(boxes):
                x0, y0, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
                if x0 <= cx < x0 + w and y0 <= cy < y0 + h:
                    if gt_areas[gi] < best_area:
                        best_area = gt_areas[gi]
                        best_gt_idx = gi

            if best_gt_idx >= 0:
                b = boxes[best_gt_idx]
                x0, y0, w, h = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
                is_positive[i] = True
                target_class[i] = int(b["class_id"])
                target_ltrb[i] = torch.tensor([
                    cx - x0, cy - y0, (x0 + w) - cx, (y0 + h) - cy,
                ])
                target_box[i] = torch.tensor([x0, y0, x0 + w, y0 + h])
                matched_gt.add(best_gt_idx)

    n_unmatched = n_gt - len(matched_gt)

    return AssignmentResult(
        is_positive=is_positive, target_class=target_class, target_ltrb=target_ltrb,
        target_box=target_box, n_gt_boxes=n_gt, n_unmatched_gt_boxes=n_unmatched,
    )
