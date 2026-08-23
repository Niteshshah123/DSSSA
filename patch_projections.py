"""
================================================================================
Patch Projection Interface (M6, item 2)
================================================================================
COMMON INTERFACE (as required): every projection strategy is a
BasePatchProjection subclass implementing forward(patches) -> (N, embedding_dim).
EDPS and the Transformer only ever see this output shape, so any projection
strategy can be swapped in via config.projection_type without touching
anything downstream.

Three implementations, differing in HOW they treat the (C, P, P) patch tensor:

  - LinearProjection: flattens (C, P, P) entirely and applies one nn.Linear.
    Cheapest, no spatial or temporal inductive bias.

  - CNNProjection: treats C as ordinary feature channels and applies 2D
    convolution purely over the (P, P) spatial dimensions, then global
    pools. Has spatial inductive bias (local pixel neighborhoods matter)
    but does NOT treat the channel axis as a temporal/depth axis.

  - Conv3DProjection: reshapes to (1, C, P, P) and applies 3D convolution
    treating C as an explicit temporal/depth axis alongside the two spatial
    axes. This is the only projection with a genuine temporal inductive
    bias baked into the projection itself (as opposed to relying solely on
    the separate temporal encoding in M6 item 4).

Registered in PROJECTION_REGISTRY so `build_patch_projection` can construct
any of them from a string, keeping patch_embedding_module.py agnostic to
which concrete class is used.
================================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BasePatchProjection(nn.Module, ABC):
    """Common interface: forward(patches: (N, C, P, P)) -> (N, embedding_dim)."""

    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearProjection(BasePatchProjection):
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int):
        super().__init__(in_channels, patch_size, embedding_dim)
        in_features = in_channels * patch_size * patch_size
        self.linear = nn.Linear(in_features, embedding_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        n = patches.shape[0]
        flat = patches.reshape(n, -1)
        return self.linear(flat)


class CNNProjection(BasePatchProjection):
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int, hidden_channels: int = 32):
        super().__init__(in_channels, patch_size, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.conv(patches)          # (N, hidden, P, P)
        x = self.pool(x).flatten(1)     # (N, hidden)
        return self.linear(x)


class Conv3DProjection(BasePatchProjection):
    def __init__(self, in_channels: int, patch_size: int, embedding_dim: int, hidden_channels: int = 16):
        super().__init__(in_channels, patch_size, embedding_dim)
        self.conv = nn.Sequential(
            nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Linear(hidden_channels, embedding_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        n = patches.shape[0]
        x = patches.unsqueeze(1)        # (N, 1, C, P, P) -- C treated as depth
        x = self.conv(x)                # (N, hidden, C, P, P)
        x = self.pool(x).flatten(1)     # (N, hidden)
        return self.linear(x)


PROJECTION_REGISTRY = {
    "linear": LinearProjection,
    "cnn": CNNProjection,
    "conv3d": Conv3DProjection,
}


def build_patch_projection(
    projection_type: str, in_channels: int, patch_size: int, embedding_dim: int, hidden_channels: int = 32,
) -> BasePatchProjection:
    if projection_type not in PROJECTION_REGISTRY:
        raise ValueError(
            f"Unknown projection_type {projection_type!r}. "
            f"Registered options: {list(PROJECTION_REGISTRY.keys())}"
        )
    cls = PROJECTION_REGISTRY[projection_type]
    if projection_type == "linear":
        return cls(in_channels, patch_size, embedding_dim)
    return cls(in_channels, patch_size, embedding_dim, hidden_channels=hidden_channels)
