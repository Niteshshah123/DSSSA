"""
================================================================================
Event-Data Augmentation (M10)
================================================================================
Applied at the raw (t, x, y, p) level, train split only, BEFORE M4
preprocessing (so M4's filtering behaves identically whether augmentation
is on or off).

INCLUDED (per design review):
  - Temporal jitter: small per-event timestamp perturbation. Physically
    plausible (sensor timing variance).
  - Event dropout: randomly drops a fraction of events. Standard in
    event-detection literature; simulates sensor/scene variability.
  - Random spatial translation: shifts all events (and GT boxes) together
    by a random offset, clipping anything that leaves the sensor bounds.
  - Random crop (mild): crops to a random sub-region covering most of the
    frame, keeping only events/boxes inside it. Kept mild (80-100% area)
    since aggressive cropping risks removing the only object in a sparse
    window.

DELIBERATELY EXCLUDED (per design review):
  - Polarity flipping: polarity encodes a real physical brightness-change
    direction tied to actual scene motion. Flipping it is not a meaningful
    invariance the way a horizontal image flip is for RGB, and would teach
    physically inconsistent event signatures.
  - Rotation: the camera is rigidly automotive-mounted. Large rotations
    would teach orientations never seen at inference and are very unlikely
    to help; omitted entirely rather than implemented-but-discouraged, to
    avoid a config knob that invites an easy-to-misuse setting.
================================================================================
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from training_config import TrainingConfig


def apply_temporal_jitter(t: np.ndarray, std_us: float, rng: np.random.Generator) -> np.ndarray:
    if std_us <= 0:
        return t
    jitter = rng.normal(0, std_us, size=t.shape)
    jittered = t.astype(np.float64) + jitter
    return np.clip(jittered, 0, None).astype(t.dtype)


def apply_event_dropout(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, drop_prob: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if drop_prob <= 0 or len(t) == 0:
        return t, x, y, p
    keep_mask = rng.random(len(t)) >= drop_prob
    return t[keep_mask], x[keep_mask], y[keep_mask], p[keep_mask]


def apply_spatial_translation(
    x: np.ndarray, y: np.ndarray, boxes: np.ndarray, height: int, width: int,
    max_frac: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x, y, boxes, keep_mask) -- keep_mask marks events still inside bounds after translation."""
    if max_frac <= 0:
        return x, y, boxes, np.ones(len(x), dtype=bool)

    dx = rng.integers(-int(max_frac * width), int(max_frac * width) + 1)
    dy = rng.integers(-int(max_frac * height), int(max_frac * height) + 1)

    new_x = x + dx
    new_y = y + dy
    keep_mask = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)

    new_boxes = boxes.copy()
    if len(new_boxes) > 0:
        new_boxes["x"] = new_boxes["x"] + dx
        new_boxes["y"] = new_boxes["y"] + dy

    return new_x, new_y, new_boxes, keep_mask


def apply_random_crop(
    x: np.ndarray, y: np.ndarray, boxes: np.ndarray, height: int, width: int,
    min_area_frac: float, rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Returns keep_mask for events inside the crop, plus the crop box
    (x0, y0, crop_w, crop_h) so the caller can also filter/shift GT boxes.
    """
    area_frac = rng.uniform(min_area_frac, 1.0)
    scale = np.sqrt(area_frac)
    crop_w = max(1, int(width * scale))
    crop_h = max(1, int(height * scale))
    x0 = rng.integers(0, max(1, width - crop_w + 1))
    y0 = rng.integers(0, max(1, height - crop_h + 1))

    keep_mask = (x >= x0) & (x < x0 + crop_w) & (y >= y0) & (y < y0 + crop_h)
    return keep_mask, x0, y0, crop_w, crop_h


def augment_sample(sample: dict, config: TrainingConfig, rng: np.random.Generator) -> dict:
    """
    sample: EDPSGen1Dataset-style dict with numpy t,x,y,p (NOT yet torch
    tensors -- augmentation happens before the M2 Dataset wraps them).
    Returns an augmented copy; never mutates the input.
    """
    if not config.use_augmentation:
        return sample

    t = np.asarray(sample["t"]).copy()
    x = np.asarray(sample["x"]).copy()
    y = np.asarray(sample["y"]).copy()
    p = np.asarray(sample["p"]).copy()
    boxes = sample["boxes"].copy()
    h, w = sample["sensor_height"], sample["sensor_width"]

    t = apply_temporal_jitter(t, config.temporal_jitter_std_us, rng)
    # Jitter can perturb relative ordering between nearby events -- M4's BAF
    # (and the parser's own validation) require strictly ascending timestamps,
    # so events must be re-sorted immediately after jittering.
    sort_order = np.argsort(t, kind="stable")
    t, x, y, p = t[sort_order], x[sort_order], y[sort_order], p[sort_order]

    t, x, y, p = apply_event_dropout(t, x, y, p, config.event_dropout_prob, rng)

    x, y, boxes, keep_mask = apply_spatial_translation(x, y, boxes, h, w, config.spatial_translation_max_frac, rng)
    t, x, y, p = t[keep_mask], x[keep_mask], y[keep_mask], p[keep_mask]
    # drop boxes that translated fully outside the frame
    if len(boxes) > 0:
        box_keep = (boxes["x"] + boxes["w"] > 0) & (boxes["x"] < w) & (boxes["y"] + boxes["h"] > 0) & (boxes["y"] < h)
        boxes = boxes[box_keep]

    crop_keep, cx0, cy0, cw, ch = apply_random_crop(x, y, boxes, h, w, config.random_crop_min_area_frac, rng)
    t, x, y, p = t[crop_keep], x[crop_keep] - cx0, y[crop_keep] - cy0, p[crop_keep]
    if len(boxes) > 0:
        boxes = boxes.copy()
        boxes["x"] = boxes["x"] - cx0
        boxes["y"] = boxes["y"] - cy0
        box_keep2 = (boxes["x"] + boxes["w"] > 0) & (boxes["x"] < cw) & (boxes["y"] + boxes["h"] > 0) & (boxes["y"] < ch)
        boxes = boxes[box_keep2]

    out = dict(sample)
    out["t"], out["x"], out["y"], out["p"] = t, x, y, p
    out["boxes"] = boxes
    out["sensor_height"], out["sensor_width"] = ch, cw
    return out
