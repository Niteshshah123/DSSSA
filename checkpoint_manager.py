"""
================================================================================
Checkpoint Manager (M10)
================================================================================
Covers requirements 2 and 3:
  - Full state capture: model components, optimizer, scheduler, GradScaler,
    epoch/global_step, best-metric-so-far, RNG state, and the exact
    TrainingConfig used -- everything needed to resume a Colab-interrupted
    run bit-for-bit, not just "continue training from these weights".
  - Best model saved to a SEPARATE file from periodic checkpoints, so
    periodic housekeeping (e.g. only keeping the last N) can never
    accidentally delete the best-known model.
  - `find_latest_checkpoint` supports automatic resume: point training at a
    directory, and it resumes from the latest periodic checkpoint if one
    exists, with no manual path-finding required.
================================================================================
"""

from __future__ import annotations

import os
import glob
import json
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from reproducibility import capture_rng_state, restore_rng_state
from training_config import TrainingConfig


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_metric: float
    best_epoch: int


def save_checkpoint(
    path: str,
    embedding_module: nn.Module,
    edps_module: nn.Module,
    encoder: nn.Module,
    detection_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    state: TrainingState,
    config: TrainingConfig,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint = {
        "embedding_module": embedding_module.state_dict(),
        "edps_module": edps_module.state_dict(),
        "encoder": encoder.state_dict(),
        "detection_head": detection_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "rng_state": capture_rng_state(),
        "config": config.to_dict(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    embedding_module: nn.Module,
    edps_module: nn.Module,
    encoder: nn.Module,
    detection_head: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
    map_location=None,
) -> TrainingState:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    embedding_module.load_state_dict(checkpoint["embedding_module"])
    edps_module.load_state_dict(checkpoint["edps_module"])
    encoder.load_state_dict(checkpoint["encoder"])
    detection_head.load_state_dict(checkpoint["detection_head"])

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng and checkpoint.get("rng_state") is not None:
        restore_rng_state(checkpoint["rng_state"])

    return TrainingState(
        epoch=checkpoint["epoch"], global_step=checkpoint["global_step"],
        best_metric=checkpoint["best_metric"], best_epoch=checkpoint["best_epoch"],
    )


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Finds the most recent PERIODIC checkpoint (never the 'best' file) for auto-resume."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not candidates:
        return None

    def epoch_num(p):
        base = os.path.basename(p)
        try:
            return int(base.replace("checkpoint_epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1

    return max(candidates, key=epoch_num)


def best_checkpoint_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "best_model.pt")


def periodic_checkpoint_path(checkpoint_dir: str, epoch: int) -> str:
    return os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")


def save_training_config_snapshot(checkpoint_dir: str, config: TrainingConfig):
    """Requirement 4: a standalone, human-readable config file for reproducibility."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "training_config.json")
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    return path
