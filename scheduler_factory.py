"""
================================================================================
LR Scheduler Factory (M10)
================================================================================
Linear warmup (over warmup_steps) followed by cosine annealing down to
min_lr_ratio * base_lr for each parameter group's OWN base LR (so EDPS's
separate LR group is scheduled proportionally, not overwritten with a
single global LR).
================================================================================
"""

from __future__ import annotations

import math

import torch

from training_config import TrainingConfig


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig, total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = config.warmup_steps if config.warmup_steps is not None else max(1, int(config.warmup_pct * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
