"""
================================================================================
M10 Training Configuration
================================================================================
Every hyperparameter discussed in the training-strategy design review is
exposed here. The configuration is serialized with checkpoints/reports for
reproducibility.

Real-run defaults are deliberately sized for the currently available
5-6 hour GPU window:
    - 20 epochs
    - batch size 8
    - gradient accumulation 4 (effective batch 32)
    - AMP enabled
    - base LR 1e-4
    - EDPS LR 3e-4
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # ---- Reproducibility ----
    seed: int = 42
    cudnn_deterministic: bool = True

    # ---- Optimizer ----
    base_lr: float = 1e-4
    edps_lr: Optional[float] = 3e-4
    weight_decay: float = 0.01

    # ---- Schedule ----
    total_epochs: int = 20
    warmup_steps: Optional[int] = None
    warmup_pct: float = 0.05
    min_lr_ratio: float = 0.01

    # ---- Batch / accumulation ----
    batch_size: int = 8
    grad_accumulation_steps: int = 4

    # ---- AMP / stability ----
    use_amp: bool = True
    grad_clip_norm: float = 1.0

    # ---- Loss weights ----
    weight_cls: float = 1.0
    weight_giou: float = 1.0
    weight_smooth_l1_aux: float = 0.5
    weight_objectness: float = 1.0
    weight_quality: float = 1.0
    lambda_sparsity: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ---- EDPS sparsity curriculum ----
    stage1_end_pct: float = 0.65
    stage2_end_pct: float = 0.85
    curriculum_start_min_ratio: float = 0.7
    curriculum_start_max_ratio: float = 0.95
    curriculum_target_min_ratio: float = 0.2
    curriculum_target_max_ratio: float = 0.6

    # ---- Validation / checkpointing / early stopping ----
    validate_every_n_epochs: int = 1
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every_n_epochs: int = 5
    early_stopping_patience: int = 18
    early_stopping_min_delta: float = 1e-4

    # ---- Augmentation (train split only) ----
    use_augmentation: bool = True
    temporal_jitter_std_us: float = 2000.0
    event_dropout_prob: float = 0.1
    spatial_translation_max_frac: float = 0.15
    random_crop_min_area_frac: float = 0.8

    def __post_init__(self):
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be positive, got {self.base_lr}")
        if self.edps_lr is not None and self.edps_lr <= 0:
            raise ValueError(f"edps_lr must be positive when set, got {self.edps_lr}")
        if not (0.0 <= self.stage1_end_pct <= self.stage2_end_pct <= 1.0):
            raise ValueError("Require 0 <= stage1_end_pct <= stage2_end_pct <= 1")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be positive")

    @property
    def effective_edps_lr(self) -> float:
        return self.edps_lr if self.edps_lr is not None else self.base_lr

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
