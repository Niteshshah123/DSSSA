"""
================================================================================
Voxelized Batch Collation (M3)
================================================================================
M2's `collate_events` deliberately returned a list of variable-length
samples, with a note that fixed-size stacking would become possible once
voxelization existed. This is that collate function.

Given a batch of EDPSGen1Dataset samples, produces:
    voxel_batch: torch.FloatTensor, shape (B, C, H, W)   -- STACKABLE, fixed-size
    boxes_list:  list of length B, each a structured ndarray (variable length --
                 this is normal and expected for object detection targets;
                 detection heads consume a list of per-image targets, not a
                 padded tensor, at every well-known framework)
    meta_list:   list of length B, small dicts with recording_stem/window_index/
                 t_start/t_end, useful for later debugging/visualization
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset
from voxel_grid import VoxelGridGenerator, VoxelGridConfig


class VoxelizedBatchCollator:
    def __init__(self, config: VoxelGridConfig = None):
        self.generator = VoxelGridGenerator(config or VoxelGridConfig())

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List, List[Dict]]:
        grids = [self.generator.generate(sample) for sample in batch]
        voxel_batch = torch.stack(grids, dim=0)

        boxes_list = [sample["boxes"] for sample in batch]
        meta_list = [
            {
                "recording_stem": sample["recording_stem"],
                "window_index": sample["window_index"],
                "t_start": sample["t_start"],
                "t_end": sample["t_end"],
            }
            for sample in batch
        ]
        return voxel_batch, boxes_list, meta_list


def build_voxel_dataloader(
    dataset: EDPSGen1Dataset,
    voxel_config: VoxelGridConfig = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = VoxelizedBatchCollator(voxel_config)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        **kwargs,
    )
