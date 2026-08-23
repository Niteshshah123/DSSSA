"""
================================================================================
M5 Patch Configuration
================================================================================
Nothing hardcoded: patch size, padding behavior, and adjacency neighborhood
mode are all configurable here. Sensor dimensions (H, W) are NOT fixed
constants in this config -- they are read from whatever voxel grid tensor is
actually passed in at runtime, so this module works unchanged if the sensor
resolution or voxel grid channel count ever changes upstream.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchConfig:
    patch_size: int = 16                    # 8 | 16 | 32, but any positive int is accepted
    padding_mode: str = "zero"              # "zero" | "ignore"
    adjacency_mode: str = "4"               # "4" | "8"

    def __post_init__(self):
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.padding_mode not in ("zero", "ignore"):
            raise ValueError(f"padding_mode must be 'zero' or 'ignore', got {self.padding_mode!r}")
        if self.adjacency_mode not in ("4", "8"):
            raise ValueError(f"adjacency_mode must be '4' or '8', got {self.adjacency_mode!r}")
