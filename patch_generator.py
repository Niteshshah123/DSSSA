"""
================================================================================
PatchGenerator (M5 orchestrator)
================================================================================
Combines patch partitioning, activity statistics, and adjacency-map
generation into one call. Produces every patch and its descriptive stats --
NO pruning happens here (that's M7/EDPS's job).
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

import torch

from patch_config import PatchConfig
from patch_metadata import PatchMetadata
from patch_partition import compute_patch_grid_dims, partition_voxel_grid
from patch_statistics import PatchActivityStats, compute_patch_activity_stats
from patch_adjacency import build_adjacency_map
from voxel_grid import VoxelGridConfig


@dataclass
class PatchGenerationResult:
    patches: torch.Tensor                    # (N, C, patch_size, patch_size)
    metadata: List[PatchMetadata]
    stats: List[PatchActivityStats]
    adjacency: Dict[int, List[int]]
    n_rows: int
    n_cols: int


class PatchGenerator:
    def __init__(self, patch_config: PatchConfig, voxel_config: VoxelGridConfig):
        self.patch_config = patch_config
        self.voxel_config = voxel_config
        self._adj_cache = {}

    def generate(self, voxel: torch.Tensor) -> PatchGenerationResult:
        c, h, w = voxel.shape
        n_rows, n_cols, _, _ = compute_patch_grid_dims(h, w, self.patch_config)

        patches, metadata = partition_voxel_grid(voxel, self.patch_config)
        stats = compute_patch_activity_stats(patches, self.voxel_config)

        key = (n_rows, n_cols)
        if key not in self._adj_cache:
            self._adj_cache[key] = build_adjacency_map(n_rows, n_cols, self.patch_config)
        adjacency = self._adj_cache[key]

        return PatchGenerationResult(
            patches=patches, metadata=metadata, stats=stats,
            adjacency=adjacency, n_rows=n_rows, n_cols=n_cols,
        )


def estimate_patch_generation_cost(
    n_patches: int, patch_size: int, temporal_dim: int, height: int, width: int,
) -> Dict[str, Any]:
    """
    Patch extraction is pure slicing/copying (no arithmetic beyond indexing),
    so FLOPs are dominated by the activity-statistics pass (sums), not the
    partitioning itself.
    """
    elements_per_patch = temporal_dim * patch_size * patch_size
    total_elements = n_patches * elements_per_patch
    return {
        "time_complexity": "O(C * H_padded * W_padded) for partitioning (one pass, pure copy); "
                            "O(C * H_padded * W_padded) for activity statistics (one pass, sums)",
        "space_complexity": "O(N_patches * C * patch_size^2), equal to the (padded) voxel grid size",
        "n_patches": n_patches,
        "elements_per_patch": elements_per_patch,
        "total_elements": total_elements,
        "estimated_flops_partitioning": total_elements,       # copy ops
        "estimated_flops_statistics": total_elements * 2,     # sum + comparison per element (pos/neg split)
        "output_bytes": total_elements * 4,  # float32
    }
