"""
================================================================================
Temporal Encoding (M6, item 4)
================================================================================
Preserves the voxel grid's temporal structure explicitly (in addition to
whatever the projection itself sees), by computing a per-bin activity
profile for each patch and encoding THAT, rather than the raw patch tensor.

Per-bin activity is computed generically for either M3 voxel grid encoding
(consistent with how M5's patch_statistics.py interprets channels):
  - "signed": one channel per bin -> activity[bin] = sum over that bin's channel
  - "separate_channels": ON bins [0:num_bins] + OFF bins [num_bins:2*num_bins]
    -> activity[bin] = sum(ON[bin]) + sum(OFF[bin])

- "learnable": a single nn.Linear(num_bins, embedding_dim) applied to the
  (normalized) activity profile.
- "sinusoidal": a FIXED (parameter-free) sinusoidal basis of shape
  (num_bins, embedding_dim); the output is the activity profile's weighted
  combination of that basis (normalized activity @ basis).
================================================================================
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def compute_temporal_activity(patches: torch.Tensor, polarity_encoding: str, num_bins: int) -> torch.Tensor:
    """
    patches: (N, C, P, P). Returns (N, num_bins) per-bin activity magnitude,
    L1-normalized per patch (so the profile reflects RELATIVE temporal
    distribution, independent of how many raw events the patch had overall
    -- overall magnitude is already captured by the projection branch).
    """
    if polarity_encoding == "signed":
        activity = patches.abs().sum(dim=(2, 3))  # (N, num_bins)
    else:
        on = patches[:, :num_bins].sum(dim=(2, 3))
        off = patches[:, num_bins:].sum(dim=(2, 3))
        activity = on + off

    totals = activity.sum(dim=1, keepdim=True)
    normalized = torch.where(totals > 0, activity / totals, torch.zeros_like(activity))
    return normalized


class TemporalEncoding(nn.Module):
    def __init__(self, num_bins: int, embedding_dim: int, mode: str):
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.mode = mode

        if mode == "learnable":
            self.proj = nn.Linear(num_bins, embedding_dim)
        elif mode == "sinusoidal":
            basis = self._build_sinusoidal_basis(num_bins, embedding_dim)
            self.register_buffer("basis", basis)  # fixed, not a learnable parameter
        else:
            raise ValueError(f"mode must be 'learnable' or 'sinusoidal', got {mode!r}")

    @staticmethod
    def _build_sinusoidal_basis(num_bins: int, embedding_dim: int) -> torch.Tensor:
        position = torch.arange(num_bins).float().unsqueeze(1)          # (num_bins, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        basis = torch.zeros(num_bins, embedding_dim)
        basis[:, 0::2] = torch.sin(position * div_term)
        n_cos = basis[:, 1::2].shape[1]
        basis[:, 1::2] = torch.cos(position * div_term[:n_cos])
        return basis

    def forward(self, activity_profile: torch.Tensor) -> torch.Tensor:
        if self.mode == "learnable":
            # nn.Linear parameters follow the module device. The input should
            # normally already be on that device, but keeping the contract
            # explicit makes the CUDA path easier to diagnose.
            if activity_profile.device != self.proj.weight.device:
                activity_profile = activity_profile.to(self.proj.weight.device)
            return self.proj(activity_profile)

        # "basis" is a registered buffer, so module.to(device) moves it with
        # the rest of the model. Align the activity tensor explicitly.
        if activity_profile.device != self.basis.device:
            activity_profile = activity_profile.to(self.basis.device)
        return activity_profile @ self.basis  # (N, num_bins) @ (num_bins, embedding_dim) -> (N, embedding_dim)
