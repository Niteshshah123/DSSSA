"""
================================================================================
Detection Loss Components (M10)
================================================================================
Per the approved design:
  - Classification: Focal Loss, computed ONLY over positive-assigned tokens
    (FCOS-style: the classification branch predicts class identity only;
    background/foreground is the objectness branch's job, avoiding the need
    for an artificial "background" class in the softmax).
  - Box regression: GIoU (primary) + optional Smooth-L1 on raw (l,t,r,b)
    (auxiliary, most useful early in training when predicted/GT boxes barely
    overlap and GIoU's gradient is less informative).
  - Objectness: BCE over ALL selected tokens (target = 1 if positive-assigned,
    else 0).
  - Quality: regressed toward the ACTUAL IoU between the predicted and
    assigned GT box (stop-gradient on the target), per VarifocalNet-style
    IoU-aware quality estimation rather than a hand-designed proxy.
================================================================================
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    logits: (M, num_classes), targets: (M,) int64 class indices.
    Standard multi-class focal loss (softmax cross-entropy reweighted by
    (1-p_t)^gamma), reduced to mean over M.
    """
    if logits.shape[0] == 0:
        return torch.tensor(0.0, device=logits.device)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - target_probs).clamp(min=1e-8) ** gamma
    loss = -alpha * focal_weight * target_log_probs
    return loss.mean()


def box_giou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    boxes_a, boxes_b: (M, 4) in (x0, y0, x1, y1) format. Returns (M,) GIoU values.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)

    inter_x0 = torch.max(boxes_a[:, 0], boxes_b[:, 0])
    inter_y0 = torch.max(boxes_a[:, 1], boxes_b[:, 1])
    inter_x1 = torch.min(boxes_a[:, 2], boxes_b[:, 2])
    inter_y1 = torch.min(boxes_a[:, 3], boxes_b[:, 3])
    inter_area = (inter_x1 - inter_x0).clamp(min=0) * (inter_y1 - inter_y0).clamp(min=0)

    union = area_a + area_b - inter_area
    iou = inter_area / union.clamp(min=1e-7)

    enclose_x0 = torch.min(boxes_a[:, 0], boxes_b[:, 0])
    enclose_y0 = torch.min(boxes_a[:, 1], boxes_b[:, 1])
    enclose_x1 = torch.max(boxes_a[:, 2], boxes_b[:, 2])
    enclose_y1 = torch.max(boxes_a[:, 3], boxes_b[:, 3])
    enclose_area = (enclose_x1 - enclose_x0).clamp(min=0) * (enclose_y1 - enclose_y0).clamp(min=0)

    giou = iou - (enclose_area - union) / enclose_area.clamp(min=1e-7)
    return giou


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.shape[0] == 0:
        return torch.tensor(0.0, device=pred_boxes.device)
    giou = box_giou(pred_boxes, target_boxes)
    return (1.0 - giou).mean()


def smooth_l1_ltrb_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    if pred_ltrb.shape[0] == 0:
        return torch.tensor(0.0, device=pred_ltrb.device)
    return F.smooth_l1_loss(pred_ltrb, target_ltrb)


def objectness_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits, targets: (M,). targets in {0.,1.}."""
    if logits.shape[0] == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets)


def quality_iou_loss(quality_logits: torch.Tensor, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """
    Regresses the quality branch toward the ACTUAL IoU between predicted and
    target boxes (stop-gradient on the IoU target -- it's a label here, not
    something we want to backprop the box-regression loss through twice).
    """
    if quality_logits.shape[0] == 0:
        return torch.tensor(0.0, device=quality_logits.device)
    with torch.no_grad():
        giou = box_giou(pred_boxes, target_boxes)
        iou_target = giou.clamp(min=0.0, max=1.0)  # GIoU can be negative; IoU proper is >= 0
    quality_pred = torch.sigmoid(quality_logits)
    return F.mse_loss(quality_pred, iou_target)
