"""
================================================================================
mAP Metric (M10)
================================================================================
Standard all-point-interpolation Average Precision, computed per class and
averaged (mAP), at IoU=0.5 (primary checkpoint-selection metric, matching
the original SRS spec) and additionally at the COCO-style 0.5:0.95 sweep.

Matching: greedy, confidence-sorted, IoU >= threshold, one prediction may
match at most one GT box and vice versa (standard COCO/VOC-style protocol).
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np


def compute_iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """pred_boxes, gt_boxes: (N,4)/(M,4) in (x0,y0,x1,y1). Returns (N,M) IoU matrix."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)))

    pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]).clip(0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clip(0)
    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]).clip(0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clip(0)

    iou = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i in range(len(pred_boxes)):
        ix0 = np.maximum(pred_boxes[i, 0], gt_boxes[:, 0])
        iy0 = np.maximum(pred_boxes[i, 1], gt_boxes[:, 1])
        ix1 = np.minimum(pred_boxes[i, 2], gt_boxes[:, 2])
        iy1 = np.minimum(pred_boxes[i, 3], gt_boxes[:, 3])
        inter = (ix1 - ix0).clip(0) * (iy1 - iy0).clip(0)
        union = pred_areas[i] + gt_areas - inter
        iou[i] = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou


def compute_ap_for_class(
    pred_boxes: List[np.ndarray], pred_scores: List[np.ndarray],
    gt_boxes: List[np.ndarray], iou_threshold: float,
) -> float:
    """
    Each of pred_boxes/pred_scores/gt_boxes is a list, one entry per image
    (window), for a SINGLE class. Returns AP (all-point interpolation).
    """
    all_scores = []
    all_tp = []
    n_gt_total = sum(len(g) for g in gt_boxes)
    if n_gt_total == 0:
        return float("nan")  # class never appears in this split -- undefined AP

    for img_idx in range(len(pred_boxes)):
        p_boxes = pred_boxes[img_idx]
        p_scores = pred_scores[img_idx]
        g_boxes = gt_boxes[img_idx]
        matched_gt = np.zeros(len(g_boxes), dtype=bool)

        order = np.argsort(-p_scores) if len(p_scores) > 0 else np.array([], dtype=int)
        iou_matrix = compute_iou_matrix(p_boxes, g_boxes) if len(p_boxes) > 0 else np.zeros((0, len(g_boxes)))

        for i in order:
            all_scores.append(p_scores[i])
            if len(g_boxes) == 0:
                all_tp.append(0)
                continue
            ious = iou_matrix[i]
            best_gt = np.argmax(ious) if len(ious) > 0 else -1
            if best_gt >= 0 and ious[best_gt] >= iou_threshold and not matched_gt[best_gt]:
                all_tp.append(1)
                matched_gt[best_gt] = True
            else:
                all_tp.append(0)

    if len(all_scores) == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = 1 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    # all-point interpolation: precision envelope is monotonically
    # non-increasing as recall decreases
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision, [0.0]])

    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i - 1]) * precision[i]
    return float(ap)


def compute_map(
    all_predictions: List[List[Dict[str, Any]]],  # per window: list of {box, class, score}
    all_ground_truths: List[np.ndarray],           # per window: structured array of GT boxes
    num_classes: int,
    iou_thresholds: List[float] = None,
) -> Dict[str, Any]:
    iou_thresholds = iou_thresholds or [0.5]

    per_class_ap_at_thresh: Dict[float, Dict[int, float]] = {}

    for thresh in iou_thresholds:
        per_class_ap = {}
        for c in range(num_classes):
            pred_boxes_c, pred_scores_c, gt_boxes_c = [], [], []
            for preds, gts in zip(all_predictions, all_ground_truths):
                p_this_class = [p for p in preds if p["class"] == c]
                pred_boxes_c.append(np.array([p["box"] for p in p_this_class]) if p_this_class else np.zeros((0, 4)))
                pred_scores_c.append(np.array([p["score"] for p in p_this_class]) if p_this_class else np.zeros((0,)))
                if len(gts) > 0:
                    g_this_class = gts[gts["class_id"] == c]
                    gt_boxes_c.append(np.stack([g_this_class["x"], g_this_class["y"],
                                                 g_this_class["x"] + g_this_class["w"],
                                                 g_this_class["y"] + g_this_class["h"]], axis=1) if len(g_this_class) else np.zeros((0, 4)))
                else:
                    gt_boxes_c.append(np.zeros((0, 4)))
            per_class_ap[c] = compute_ap_for_class(pred_boxes_c, pred_scores_c, gt_boxes_c, thresh)
        per_class_ap_at_thresh[thresh] = per_class_ap

    map_at_05 = float(np.nanmean(list(per_class_ap_at_thresh.get(0.5, {}).values()))) if 0.5 in per_class_ap_at_thresh else None
    map_over_thresholds = [np.nanmean(list(v.values())) for v in per_class_ap_at_thresh.values()]
    map_05_95 = float(np.nanmean(map_over_thresholds)) if len(iou_thresholds) > 1 else None

    return {
        "map_50": map_at_05,
        "map_50_95": map_05_95,
        "per_class_ap_50": per_class_ap_at_thresh.get(0.5, {}),
        "per_class_ap_all_thresholds": per_class_ap_at_thresh,
    }
