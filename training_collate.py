"""
================================================================================
Training Collate (M10)
================================================================================
Realizes the complete training-time pipeline:

    Raw Events -> [augmentation, train split only] -> M4 -> M3 -> M5 -> M6
    -> M7 (EDPS) -> GATED embeddings (the M10 wiring fix) -> M8 -> M9
    -> target assignment (for loss computation in the training loop)

Distinct from every earlier collator in two ways:
  1. Applies augmentation BEFORE M4, train split only (never val/test, so
     evaluation numbers stay comparable across epochs).
  2. Feeds the Transformer GATED embeddings (selected_embeddings * their own
     soft gate value) rather than raw selected_embeddings, so detection-loss
     gradients reach EDPS's scoring network (see gated_token_wiring.py).

Returns everything the training loop needs to compute the full loss:
  raw detection predictions, per-sample assignment results, per-sample
  EDPS sparsity-relevant importance_scores, and the recall-ceiling
  diagnostic (n_unmatched_gt_boxes) per sample.
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_config import PatchConfig
from patch_generator import PatchGenerator
from patch_embedding_module import PatchEmbeddingModule
from edps_module import EDPSModule
from gated_token_wiring import compute_gated_selected_embeddings
from token_padding import pad_token_sequences
from transformer_encoder import TransformerEncoder
from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead
from target_assignment import assign_targets
from training_config import TrainingConfig
from augmentation import augment_sample


class TrainingPipelineCollator:
    def __init__(
        self,
        dataset_root: str,
        n_rows: int,
        n_cols: int,
        embedding_module: PatchEmbeddingModule,
        edps_module: EDPSModule,
        encoder: TransformerEncoder,
        detection_head: SparseDetectionHead,
        training_config: TrainingConfig,
        preprocessing_config: PreprocessingConfig = None,
        voxel_config: VoxelGridConfig = None,
        patch_config: PatchConfig = None,
        detection_config: DetectionHeadConfig = None,
        is_train: bool = True,
        augmentation_seed: int = 0,
        device: str = "cpu",
    ):
        self.dataset_root = dataset_root
        if preprocessing_config is None:
            # Disable CPU-side O(N^2) noise searching during training for maximum GPU throughput
            preprocessing_config = PreprocessingConfig(
                use_baf=False,
                use_hot_pixel_filter=False,
                use_isolated_event_filter=False,
                use_normalization=True,
            )
        self.preprocessor = EventPreprocessor(preprocessing_config)
        self.voxel_config = voxel_config or VoxelGridConfig()
        self.voxel_generator = VoxelGridGenerator(self.voxel_config)
        self.patch_config = patch_config or PatchConfig()
        self.patch_generator = PatchGenerator(self.patch_config, self.voxel_config)

        self.embedding_module = embedding_module
        self.edps_module = edps_module
        self.encoder = encoder
        self.detection_head = detection_head
        self.detection_config = detection_config or DetectionHeadConfig()

        self.training_config = training_config
        self.is_train = is_train
        self.device = torch.device(device)
        self._aug_rng = np.random.default_rng(augmentation_seed)

    def __call__(self, batch: List[Dict[str, Any]]):
        selected_embeddings_list = []
        selected_metadata_list = []
        boxes_list = []
        meta_list = []
        raw_boxes_for_map = []

        for sample in batch:
            if self.is_train and self.training_config.use_augmentation:
                sample = augment_sample(sample, self.training_config, self._aug_rng)

            cleaned = self.preprocessor.process(sample, dataset_root=self.dataset_root)
            grid = self.voxel_generator.generate({
                "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
                "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
            })
            patch_result = self.patch_generator.generate(grid)
            patches = patch_result.patches.to(self.device, non_blocking=True)
            embeddings, embedding_meta = self.embedding_module(patches, patch_result.metadata)
            edps_out = self.edps_module(
                embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency,
            )

            gated = compute_gated_selected_embeddings(edps_out)
            selected_embeddings_list.append(gated)
            selected_metadata_list.append(edps_out.selected_metadata)
            boxes_list.append(cleaned["boxes"])
            raw_boxes_for_map.append(cleaned["boxes"])
            meta_list.append({
                "recording_stem": cleaned["recording_stem"], "window_index": cleaned["window_index"],
                "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
                "importance_scores": edps_out.importance_scores,   # full-N, for sparsity loss
                "token_reduction_stats": edps_out.token_reduction_stats,
            })

        padded, attention_mask = pad_token_sequences(selected_embeddings_list)

        # pad_token_sequences creates new tensors on CPU by default.
        # Move both tensors to the same device as the trainable pipeline
        # before passing them to the Transformer.
        padded = padded.to(self.device, non_blocking=True)
        attention_mask = attention_mask.to(self.device, non_blocking=True)

        encoded = self.encoder(
            padded,
            attention_mask=attention_mask,
            return_attention=False,
        )
        raw_predictions = self.detection_head(encoded)

        assignments = [
            assign_targets(selected_metadata_list[i], boxes_list[i])
            for i in range(len(batch))
        ]

        return {
            "raw_predictions": raw_predictions,
            "attention_mask": attention_mask,
            "assignments": assignments,
            "selected_metadata": selected_metadata_list,
            "boxes": raw_boxes_for_map,
            "meta": meta_list,
        }


def build_training_dataloader(
    dataset: EDPSGen1Dataset,
    dataset_root: str,
    n_rows: int,
    n_cols: int,
    embedding_module: PatchEmbeddingModule,
    edps_module: EDPSModule,
    encoder: TransformerEncoder,
    detection_head: SparseDetectionHead,
    training_config: TrainingConfig,
    is_train: bool = True,
    shuffle: bool = None,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    from reproducibility import worker_init_fn

    target_device = kwargs.pop("device", "cpu")
    num_workers = 0  # Collator owns autograd nn.Modules; must remain 0 to avoid process boundary errors

    collator = TrainingPipelineCollator(
        dataset_root, n_rows, n_cols, embedding_module, edps_module, encoder, detection_head,
        training_config, is_train=is_train, augmentation_seed=training_config.seed,
        device=target_device,
    )
    shuffle = is_train if shuffle is None else shuffle

    return DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collator,
        **kwargs,
    )
