"""
================================================================================
Reproducibility Utilities (M10)
================================================================================
Covers requirement 9 (seed handling) and the RNG-state portion of
requirement 2 (checkpoint resume must restore RNG state exactly, not just
re-seed from the original seed, since the RNG has advanced during training).
================================================================================
"""

from __future__ import annotations

import os
import random
from typing import Dict, Any

import numpy as np
import torch


def seed_everything(seed: int, cudnn_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    """
    Explicit DataLoader worker seeding. Forked workers do NOT automatically
    inherit reproducible RNG state from the main process across platforms/
    Python versions, so this must be passed explicitly to DataLoader(...,
    worker_init_fn=worker_init_fn) for training-data reproducibility.
    """
    base_seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
