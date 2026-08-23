"""
================================================================================
Full Pipeline Batch Collation: M4 -> M3 -> M5 -> M6 -> M7
================================================================================
Realizes the complete pipeline through EDPS. Per-sample K can legitimately
differ (that's the whole point of EDPS), so unlike every earlier collator,
this one does NOT stack selected embeddings into a single dense tensor --
it returns a LIST of per-sample EDPSOutput objects, exactly like M2's
original variable-length collate_events did before voxelization made
stacking possible. Fixed-size stacking (via padding or otherwise) is a
concern for M8's Transformer input handling, not for M7 itself.
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_config import PatchConfig
from patch_generator import PatchGenerator
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule
from edps_config import EDPSConfig
from edps_module import EDPSModule


class FullPipelineEDPSCollator:
    def __init__(
        self,
        dataset_root: str,
        n_rows: int,
        n_cols: int,
        preprocessing_config: PreprocessingConfig = None,
        voxel_config: VoxelGridConfig = None,
        patch_config: PatchConfig = None,
        embedding_config: PatchEmbeddingConfig = None,
        edps_config: EDPSConfig = None,
    ):
        self.dataset_root = dataset_root
        self.preprocessor = EventPreprocessor(preprocessing_config or PreprocessingConfig())
        self.voxel_config = voxel_config or VoxelGridConfig()
        self.voxel_generator = VoxelGridGenerator(self.voxel_config)
        self.patch_config = patch_config or PatchConfig()
        self.patch_generator = PatchGenerator(self.patch_config, self.voxel_config)

        self.embedding_config = embedding_config or PatchEmbeddingConfig()
        self.embedding_module = PatchEmbeddingModule(
            self.embedding_config, in_channels=self.voxel_config.num_channels,
            patch_size=self.patch_config.patch_size, n_rows=n_rows, n_cols=n_cols,
            polarity_encoding=self.voxel_config.polarity_encoding,
        )

        self.edps_config = edps_config or EDPSConfig()
        self.edps_module = EDPSModule(
            self.edps_config, embedding_dim=self.embedding_config.embedding_dim,
            num_bins=self.voxel_config.num_bins,
        )

    def __call__(self, batch: List[Dict[str, Any]]):
        results = []
        for sample in batch:
            cleaned = self.preprocessor.process(sample, dataset_root=self.dataset_root)
            grid = self.voxel_generator.generate({
                "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
            })
            patch_result = self.patch_generator.generate(grid)
            embeddings, embedding_meta = self.embedding_module(patch_result.patches, patch_result.metadata)
            edps_out = self.edps_module(
                embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency,
            )
            results.append({
                "edps_output": edps_out,
                "boxes": cleaned["boxes"],
                "recording_stem": cleaned["recording_stem"],
                "window_index": cleaned["window_index"],
                "n_rows": patch_result.n_rows,
                "n_cols": patch_result.n_cols,
                "patch_metadata": patch_result.metadata,
            })
        return results


def build_full_pipeline_edps_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    n_rows: int,
    n_cols: int,
    preprocessing_config: PreprocessingConfig = None,
    voxel_config: VoxelGridConfig = None,
    patch_config: PatchConfig = None,
    embedding_config: PatchEmbeddingConfig = None,
    edps_config: EDPSConfig = None,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = FullPipelineEDPSCollator(
        dataset_root, n_rows, n_cols, preprocessing_config, voxel_config,
        patch_config, embedding_config, edps_config,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, **kwargs,
    )
