"""
================================================================================
Preprocessed + Voxelized Batch Collation (M4 <-> M3 integration)
================================================================================
Realizes the approved data flow:

    Raw Event Stream -> M4 Preprocessing -> M3 Voxel Grid -> (batch stack)

Given a batch of raw EDPSGen1Dataset samples, this collator:
  1. Runs EventPreprocessor.process() on each sample (BAF -> hot-pixel ->
     isolated-event removal -> normalization)
  2. Feeds the FILTERED INTEGER (t, x, y, p) into VoxelGridGenerator
     (never the normalized floats -- see event_normalization.py docstring)
  3. Stacks the resulting fixed-size voxel grids into one batch tensor

Also aggregates per-stage StageStats across the whole batch, so a training
loop could log "how much was removed this batch" cheaply if desired.
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator


class PreprocessedVoxelizedBatchCollator:
    def __init__(
        self,
        dataset_root: str,
        preprocessing_config: PreprocessingConfig = None,
        voxel_config: VoxelGridConfig = None,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = EventPreprocessor(preprocessing_config or PreprocessingConfig())
        self.voxel_generator = VoxelGridGenerator(voxel_config or VoxelGridConfig())

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List, List[Dict], List]:
        grids = []
        boxes_list = []
        meta_list = []
        all_stage_stats = []

        for sample in batch:
            cleaned = self.preprocessor.process(sample, dataset_root=self.dataset_root)

            grid = self.voxel_generator.generate({
                "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
            })
            grids.append(grid)
            boxes_list.append(cleaned["boxes"])
            meta_list.append({
                "recording_stem": cleaned["recording_stem"],
                "window_index": cleaned["window_index"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
            })
            all_stage_stats.append(cleaned["stage_stats"])

        voxel_batch = torch.stack(grids, dim=0)
        return voxel_batch, boxes_list, meta_list, all_stage_stats


def build_preprocessed_voxel_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    preprocessing_config: PreprocessingConfig = None,
    voxel_config: VoxelGridConfig = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = PreprocessedVoxelizedBatchCollator(dataset_root, preprocessing_config, voxel_config)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, **kwargs,
    )
