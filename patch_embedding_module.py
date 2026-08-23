"""
================================================================================
PatchEmbeddingModule (M6 orchestrator, optimized)
================================================================================
Combines:
  1. Patch projection
  2. Positional encoding
  3. Temporal encoding

The mathematical computation and public output interface are unchanged.

Performance optimizations:
  - Avoids repeated positional-info construction.
  - Avoids GPU -> CPU synchronization caused by converting every positional
    and temporal embedding vector to Python lists on every training sample.
  - Keeps only the scalar positional metadata needed by M7/EDPS in the normal
    training path.
  - Optional `store_encoding_vectors=True` restores the old diagnostic behavior
    when full vectors are explicitly required.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from patch_embedding_config import PatchEmbeddingConfig
from patch_projections import build_patch_projection
from positional_encoding import PositionalEncoding
from temporal_encoding import TemporalEncoding, compute_temporal_activity
from patch_metadata import PatchMetadata


@dataclass
class EmbeddingMetadata:
    patch_index: int
    row_index: int
    col_index: int
    norm_row: float
    norm_col: float
    positional_encoding: List[float]
    temporal_encoding: Optional[List[float]]

    def to_dict(self):
        return self.__dict__.copy()


class PatchEmbeddingModule(nn.Module):
    def __init__(
        self,
        config: PatchEmbeddingConfig,
        in_channels: int,
        patch_size: int,
        n_rows: int,
        n_cols: int,
        polarity_encoding: str = "signed",
        store_encoding_vectors: bool = False,
    ):
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.polarity_encoding = polarity_encoding
        self.num_bins = (
            in_channels if polarity_encoding == "signed" else in_channels // 2
        )

        # Full encoding vectors are not consumed by M7. They used to cause a
        # very expensive GPU -> CPU synchronization through .tolist() for
        # every patch of every sample. Keep them available only when explicitly
        # requested for diagnostics/visualization.
        self.store_encoding_vectors = bool(store_encoding_vectors)

        self.projection = build_patch_projection(
            config.projection_type,
            in_channels,
            patch_size,
            config.embedding_dim,
            hidden_channels=config.cnn_hidden_channels,
        )

        self.positional = PositionalEncoding(
            config.embedding_dim,
            config.positional_encoding_type,
            n_rows,
            n_cols,
        )

        self.temporal: Optional[TemporalEncoding] = None
        if config.use_temporal_encoding:
            self.temporal = TemporalEncoding(
                self.num_bins,
                config.embedding_dim,
                config.temporal_encoding_type,
            )

    def forward(
        self,
        patches: torch.Tensor,
        metadata: List[PatchMetadata],
    ):
        """
        patches: (N, C, P, P)
        metadata: list of PatchMetadata, length N.

        Returns:
            embeddings: (N, embedding_dim)
            meta_out: list[EmbeddingMetadata]
        """
        n = patches.shape[0]

        if n != len(metadata):
            raise ValueError(
                f"patches has {n} entries but metadata has "
                f"{len(metadata)} -- they must correspond 1:1."
            )

        # Python extraction of two integer lists is cheap compared with the
        # GPU projection and, unlike the old .tolist() path below, does not
        # synchronize GPU tensors with the CPU.
        row_indices = [m.row_index for m in metadata]
        col_indices = [m.col_index for m in metadata]

        # Main learned projection remains exactly unchanged.
        patch_embed = self.projection(patches)

        # Positional encoding is now cached/tensorized by PositionalEncoding.
        pos_embed = self.positional(row_indices, col_indices)
        combined = patch_embed + pos_embed

        temporal_embed = None
        if self.temporal is not None:
            activity = compute_temporal_activity(
                patches,
                self.polarity_encoding,
                self.num_bins,
            )
            temporal_embed = self.temporal(activity)
            combined = combined + temporal_embed

        # M7 only consumes row/column and normalized coordinates from
        # EmbeddingMetadata. Therefore do NOT transfer every D-dimensional
        # positional/temporal vector from GPU to CPU during normal training.
        #
        # This is the critical performance fix: the old implementation did
        # pos_embed[i].detach().tolist() for every patch, forcing a synchronous
        # device-to-host copy for thousands of patches per epoch.
        pos_info = self.positional.compute_positional_info(
            row_indices, col_indices
        )

        if self.store_encoding_vectors:
            # Explicit diagnostic mode: preserve the old behavior.
            pos_vectors = pos_embed.detach().cpu().tolist()
            temporal_vectors = (
                temporal_embed.detach().cpu().tolist()
                if temporal_embed is not None
                else None
            )
        else:
            pos_vectors = None
            temporal_vectors = None

        meta_out: List[EmbeddingMetadata] = []
        for i, (m, pinfo) in enumerate(zip(metadata, pos_info)):
            meta_out.append(
                EmbeddingMetadata(
                    patch_index=m.patch_index,
                    row_index=m.row_index,
                    col_index=m.col_index,
                    norm_row=pinfo.norm_row,
                    norm_col=pinfo.norm_col,
                    positional_encoding=(
                        pos_vectors[i] if pos_vectors is not None else []
                    ),
                    temporal_encoding=(
                        temporal_vectors[i]
                        if temporal_vectors is not None
                        else None
                    ),
                )
            )

        return combined, meta_out


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
