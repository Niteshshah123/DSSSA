"""
================================================================================
Full Pipeline Batch Collation: M4 -> M3 -> M5 -> M6 -> M7 -> M8
================================================================================
Realizes the complete pipeline through the Transformer encoder. EDPS's
variable per-sample K is handled here via padding + attention mask, so the
Transformer itself never needs to know about the ragged structure upstream.
No detection head is applied -- output is still (B, K_max, embedding_dim)
encoded tokens plus the mask, boxes, and metadata needed by a future head.
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
from edps_config import EDPSConfig
from edps_module import EDPSModule
from token_padding import pad_token_sequences
from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder


class FullPipelineTransformerCollator:
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
        transformer_config: TransformerEncoderConfig = None,
        return_attention: bool = False,
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

        self.transformer_config = transformer_config or TransformerEncoderConfig()
        self.encoder = TransformerEncoder(self.transformer_config)
        self.return_attention = return_attention

    def __call__(self, batch: List[Dict[str, Any]]):
        selected_embeddings_list = []
        boxes_list = []
        meta_list = []
        selected_metadata_list = []

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

            selected_embeddings_list.append(edps_out.selected_embeddings)
            selected_metadata_list.append(edps_out.selected_metadata)
            boxes_list.append(cleaned["boxes"])
            meta_list.append({
                "recording_stem": cleaned["recording_stem"],
                "window_index": cleaned["window_index"],
            })

        padded, attention_mask = pad_token_sequences(selected_embeddings_list)

        if self.return_attention:
            encoded, attn_maps = self.encoder(padded, attention_mask=attention_mask, return_attention=True)
        else:
            encoded, attn_maps = self.encoder(padded, attention_mask=attention_mask, return_attention=False), None

        return {
            "encoded_tokens": encoded,
            "attention_mask": attention_mask,
            "attention_maps": attn_maps,
            "boxes": boxes_list,
            "meta": meta_list,
            "selected_metadata": selected_metadata_list,
        }


def build_full_pipeline_transformer_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    n_rows: int,
    n_cols: int,
    preprocessing_config: PreprocessingConfig = None,
    voxel_config: VoxelGridConfig = None,
    patch_config: PatchConfig = None,
    embedding_config: PatchEmbeddingConfig = None,
    edps_config: EDPSConfig = None,
    transformer_config: TransformerEncoderConfig = None,
    return_attention: bool = False,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    collator = FullPipelineTransformerCollator(
        dataset_root, n_rows, n_cols, preprocessing_config, voxel_config,
        patch_config, embedding_config, edps_config, transformer_config, return_attention,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collator, **kwargs,
    )
