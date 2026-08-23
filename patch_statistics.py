from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from voxel_grid import VoxelGridConfig


@dataclass
class PatchActivityStats:
    patch_index: int
    total_event_count: float
    positive_event_count: float
    negative_event_count: float
    event_density: float
    temporal_activity_distribution: List[float] = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def compute_patch_activity_stats(
    patches: torch.Tensor, voxel_config: VoxelGridConfig,
) -> List[PatchActivityStats]:
    """
    Vectorized patch statistics.

    The returned Python objects preserve the existing M5/M7 interface.
    """
    n, c, ph, pw = patches.shape
    num_bins = voxel_config.num_bins

    if voxel_config.polarity_encoding == "signed":
        positive = patches.clamp_min(0)
        negative = (-patches.clamp_max(0))
        pos_sums = positive.sum(dim=(1, 2, 3))
        neg_sums = negative.sum(dim=(1, 2, 3))
        temporal_dists = patches.sum(dim=(2, 3))
    else:
        on = patches[:, :num_bins]
        off = patches[:, num_bins:]
        pos_sums = on.sum(dim=(1, 2, 3))
        neg_sums = off.sum(dim=(1, 2, 3))
        temporal_dists = on.sum(dim=2).sum(dim=2) + off.sum(dim=2).sum(dim=2)

    totals = pos_sums + neg_sums
    densities = totals / float(ph * pw)

    # One host synchronization instead of four independent tensor->list
    # conversions. This matters when M5 is ever fed CUDA tensors.
    packed = torch.cat(
        [
            pos_sums.reshape(n, 1),
            neg_sums.reshape(n, 1),
            totals.reshape(n, 1),
            densities.reshape(n, 1),
            temporal_dists,
        ],
        dim=1,
    ).detach().cpu().tolist()

    rows = num_bins + 4
    return [
        PatchActivityStats(
            patch_index=i,
            total_event_count=float(packed[i][2]),
            positive_event_count=float(packed[i][0]),
            negative_event_count=float(packed[i][1]),
            event_density=float(packed[i][3]),
            temporal_activity_distribution=[
                float(v) for v in packed[i][4:4 + num_bins]
            ],
        )
        for i in range(n)
    ]
