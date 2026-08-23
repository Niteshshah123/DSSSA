"""
================================================================================
Full Pipeline Batch Collation: M4 -> M3 -> M5 -> M6
================================================================================
Realizes: Raw Events -> M4 Preprocessing -> M3 Voxel Grid -> M5 Patch
Generation -> M6 Patch Embedding.

Since M5's patch grid layout is identical for every window (fixed sensor
resolution + fixed patch_size), the embedding module's positional/temporal
encoding submodules (which depend on n_rows/n_cols/num_bins) are constructed
ONCE and reused across the whole dataset, not rebuilt per sample.

Output embeddings ARE stackable across a batch: (B, N_patches, embedding_dim).
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_config import PatchConfig
from patch_generator import PatchGenerator
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule


class FullPipelineEmbeddingCollator:
    def __init__(
        self,
        dataset_root: str,
        n_rows: int,
        n_cols: int,
        preprocessing_config: PreprocessingConfig = None,
        voxel_config: VoxelGridConfig = None,
        patch_config: PatchConfig = None,
        embedding_config: PatchEmbeddingConfig = None,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = EventPreprocessor(preprocessing_config or PreprocessingConfig())
        self.voxel_config = voxel_config or VoxelGridConfig()
        self.voxel_generator = VoxelGridGenerator(self.voxel_config)
        self.patch_config = patch_config or PatchConfig()
        self.patch_generator = PatchGenerator(self.patch_config, self.voxel_config)

        embedding_config = embedding_config or PatchEmbeddingConfig()
        self.embedding_module = PatchEmbeddingModule(
            embedding_config, in_channels=self.voxel_config.num_channels,
            patch_size=self.patch_config.patch_size, n_rows=n_rows, n_cols=n_cols,
            polarity_encoding=self.voxel_config.polarity_encoding,
        )

    def __call__(self, batch: List[Dict[str, Any]]):
        embeddings_batch = []
        boxes_list = []
        meta_list = []
        shared_patch_metadata = None
        shared_adjacency = None
        embedding_meta_out = None

        for sample in batch:
            cleaned = self.preprocessor.process(sample, dataset_root=self.dataset_root)
            grid = self.voxel_generator.generate({
                "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
            })
            patch_result = self.patch_generator.generate(grid)
            embeddings, meta_out = self.embedding_module(patch_result.patches, patch_result.metadata)

            embeddings_batch.append(embeddings)
            boxes_list.append(cleaned["boxes"])
            meta_list.append({
                "recording_stem": cleaned["recording_stem"],
                "window_index": cleaned["window_index"],
            })

            if shared_patch_metadata is None:
                shared_patch_metadata = patch_result.metadata
                shared_adjacency = patch_result.adjacency
                embedding_meta_out = meta_out

        embeddings_tensor = torch.stack(embeddings_batch, dim=0)  # (B, N_patches, embedding_dim)
        return {
            "embeddings": embeddings_tensor,
            "boxes": boxes_list,
            "meta": meta_list,
            "patch_metadata": shared_patch_metadata,
            "adjacency": shared_adjacency,
            "embedding_metadata": embedding_meta_out,
        }


def build_full_pipeline_embedding_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    n_rows: int,
    n_cols: int,
    preprocessing_config: PreprocessingConfig = None,
    voxel_config: VoxelGridConfig = None,
    patch_config: PatchConfig = None,
    embedding_config: PatchEmbeddingConfig = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = FullPipelineEmbeddingCollator(
        dataset_root, n_rows, n_cols, preprocessing_config, voxel_config, patch_config, embedding_config,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, **kwargs,
    )
