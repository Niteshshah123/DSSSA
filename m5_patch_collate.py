"""
================================================================================
Full Pipeline Batch Collation: M4 -> M3 -> M5
================================================================================
Realizes: Raw Events -> M4 Preprocessing -> M3 Voxel Grid -> M5 Patch Generation

Since every voxel grid in the dataset has the same (C, H, W) shape (fixed
sensor resolution + fixed config), the patch grid layout (n_rows, n_cols,
N_patches) is IDENTICAL for every window. This means patch tensors ARE
stackable across a batch: (B, N_patches, C, patch_size, patch_size).

Metadata and adjacency are identical for every sample in a dataset that
shares config (patch layout depends only on H, W, patch_size -- never on
event content), so they're returned once per batch, not duplicated per
sample.
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
from patch_config import PatchConfig
from patch_generator import PatchGenerator


class FullPipelineBatchCollator:
    def __init__(
        self,
        dataset_root: str,
        preprocessing_config: PreprocessingConfig = None,
        voxel_config: VoxelGridConfig = None,
        patch_config: PatchConfig = None,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = EventPreprocessor(preprocessing_config or PreprocessingConfig())
        self.voxel_config = voxel_config or VoxelGridConfig()
        self.voxel_generator = VoxelGridGenerator(self.voxel_config)
        self.patch_config = patch_config or PatchConfig()
        self.patch_generator = PatchGenerator(self.patch_config, self.voxel_config)

    def __call__(self, batch: List[Dict[str, Any]]):
        patch_batches = []
        boxes_list = []
        meta_list = []
        shared_patch_metadata = None
        shared_adjacency = None
        n_rows = n_cols = None

        for sample in batch:
            cleaned = self.preprocessor.process(sample, dataset_root=self.dataset_root)
            grid = self.voxel_generator.generate({
                "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
            })
            result = self.patch_generator.generate(grid)

            patch_batches.append(result.patches)
            boxes_list.append(cleaned["boxes"])
            meta_list.append({
                "recording_stem": cleaned["recording_stem"],
                "window_index": cleaned["window_index"],
            })

            if shared_patch_metadata is None:
                shared_patch_metadata = result.metadata
                shared_adjacency = result.adjacency
                n_rows, n_cols = result.n_rows, result.n_cols

        patch_tensor_batch = torch.stack(patch_batches, dim=0)  # (B, N_patches, C, ps, ps)
        return {
            "patches": patch_tensor_batch,
            "boxes": boxes_list,
            "meta": meta_list,
            "patch_metadata": shared_patch_metadata,
            "adjacency": shared_adjacency,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }


def build_full_pipeline_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    preprocessing_config: PreprocessingConfig = None,
    voxel_config: VoxelGridConfig = None,
    patch_config: PatchConfig = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = FullPipelineBatchCollator(dataset_root, preprocessing_config, voxel_config, patch_config)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, **kwargs,
    )
