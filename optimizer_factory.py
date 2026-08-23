"""
================================================================================
Optimizer Factory (M10)
================================================================================
AdamW with parameter groups:
  1. Weight-decay applied to Linear/Conv weights.
  2. Weight-decay EXCLUDED for biases and LayerNorm parameters (a
     commonly-missed detail; decaying norm/bias parameters has little
     regularization benefit and can hurt training stability).
  3. EDPS's scoring network gets its OWN learning rate (requirement 1:
     configurable via TrainingConfig.edps_lr, defaults to base_lr if None
     -- i.e. "same LR for all modules" is the default, per your requirement,
     with a documented path to experiment with a higher EDPS LR later).
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn

from training_config import TrainingConfig


def _split_decay_params(module: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_optimizer(
    embedding_module: nn.Module,
    edps_module: nn.Module,
    encoder: nn.Module,
    detection_head: nn.Module,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    param_groups: List[Dict[str, Any]] = []

    # EDPS gets its own (configurable) LR group
    edps_decay, edps_no_decay = _split_decay_params(edps_module)
    param_groups.append({"params": edps_decay, "lr": config.effective_edps_lr, "weight_decay": config.weight_decay})
    param_groups.append({"params": edps_no_decay, "lr": config.effective_edps_lr, "weight_decay": 0.0})

    # Everything else (embedding, encoder, detection head) uses base_lr
    for m in (embedding_module, encoder, detection_head):
        decay, no_decay = _split_decay_params(m)
        param_groups.append({"params": decay, "lr": config.base_lr, "weight_decay": config.weight_decay})
        param_groups.append({"params": no_decay, "lr": config.base_lr, "weight_decay": 0.0})

    return torch.optim.AdamW(param_groups)
