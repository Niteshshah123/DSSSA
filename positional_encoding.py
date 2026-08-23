"""
================================================================================
Positional Encoding (M6, optimized)
================================================================================
Produces a (N_patches, embedding_dim) positional embedding from each patch's
grid position.

Optimization changes:
  - Avoids Python list comprehensions for sinusoidal coordinates.
  - Caches the fixed sinusoidal frequency terms.
  - Caches the complete fixed grid encoding for the common full-grid case.
  - Keeps the original learnable row/column embedding behavior.
  - Keeps compute_positional_info() and the public forward() interface
    backward-compatible.

The mathematical positional encoding is unchanged.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn


@dataclass
class PositionalInfo:
    patch_index: int
    row_index: int
    col_index: int
    norm_row: float
    norm_col: float


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, mode: str, n_rows: int, n_cols: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.n_rows = n_rows
        self.n_cols = n_cols

        if mode == "learnable":
            self.row_embed = nn.Embedding(n_rows, embedding_dim)
            self.col_embed = nn.Embedding(n_cols, embedding_dim)

        elif mode == "sinusoidal":
            self.register_buffer(
                "_device_anchor",
                torch.empty(0),
                persistent=False,
            )

            # Fixed frequencies are created once instead of on every forward.
            half = embedding_dim // 2
            row_dim = half
            col_dim = embedding_dim - half

            row_div = torch.exp(
                torch.arange(0, row_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / row_dim)
            ) if row_dim else torch.empty(0)

            col_div = torch.exp(
                torch.arange(0, col_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / col_dim)
            ) if col_dim else torch.empty(0)

            self.register_buffer("_row_div_term", row_div, persistent=False)
            self.register_buffer("_col_div_term", col_div, persistent=False)

            # Precompute the complete sensor patch-grid encoding. This is
            # parameter-free and therefore safe to cache; the cache is moved
            # with the module when .to(device) is called.
            rows = torch.arange(n_rows, dtype=torch.float32)
            cols = torch.arange(n_cols, dtype=torch.float32)

            norm_rows = (
                rows / (n_rows - 1) if n_rows > 1 else torch.zeros_like(rows)
            )
            norm_cols = (
                cols / (n_cols - 1) if n_cols > 1 else torch.zeros_like(cols)
            )

            row_enc = self._sinusoidal_1d_cached(
                norm_rows, row_dim, row_div
            )
            col_enc = self._sinusoidal_1d_cached(
                norm_cols, col_dim, col_div
            )

            # Grid order is row-major: patch index = row*n_cols + col.
            full = (
                row_enc[:, None, :] + torch.zeros(
                    (1, n_cols, row_dim), dtype=row_enc.dtype
                )
            )
            full = torch.cat(
                [
                    row_enc[:, None, :].expand(n_rows, n_cols, row_dim),
                    col_enc[None, :, :].expand(n_rows, n_cols, col_dim),
                ],
                dim=-1,
            ).reshape(n_rows * n_cols, embedding_dim)

            self.register_buffer(
                "_full_grid_encoding",
                full,
                persistent=False,
            )

        else:
            raise ValueError(
                f"mode must be 'learnable' or 'sinusoidal', got {mode!r}"
            )

    @staticmethod
    def _sinusoidal_1d_cached(
        positions: torch.Tensor,
        dim: int,
        div_term: torch.Tensor,
    ) -> torch.Tensor:
        if dim == 0:
            return positions.new_empty((positions.shape[0], 0))
        if dim % 2 != 0:
            raise ValueError(
                f"sinusoidal encoding dimension must be even, got {dim}"
            )

        angles = positions.unsqueeze(-1) * div_term
        enc = torch.empty(
            positions.shape[0],
            dim,
            dtype=positions.dtype,
            device=positions.device,
        )
        enc[:, 0::2] = torch.sin(angles)
        enc[:, 1::2] = torch.cos(angles)
        return enc

    def compute_positional_info(
        self,
        row_indices: Sequence[int],
        col_indices: Sequence[int],
    ) -> List[PositionalInfo]:
        # This method is primarily metadata construction, so keeping the
        # Python objects here is intentional and preserves the public API.
        denom_r = self.n_rows - 1
        denom_c = self.n_cols - 1

        return [
            PositionalInfo(
                patch_index=i,
                row_index=int(r),
                col_index=int(c),
                norm_row=(float(r) / denom_r) if denom_r > 0 else 0.0,
                norm_col=(float(c) / denom_c) if denom_c > 0 else 0.0,
            )
            for i, (r, c) in enumerate(zip(row_indices, col_indices))
        ]

    def forward(
        self,
        row_indices: Sequence[int],
        col_indices: Sequence[int],
    ) -> torch.Tensor:
        if len(row_indices) != len(col_indices):
            raise ValueError("row_indices and col_indices must have equal length")

        if self.mode == "learnable":
            device = self.row_embed.weight.device
            rows = torch.as_tensor(
                row_indices, dtype=torch.long, device=device
            )
            cols = torch.as_tensor(
                col_indices, dtype=torch.long, device=device
            )
            return self.row_embed(rows) + self.col_embed(cols)

        device = self._device_anchor.device

        # Fast path for the normal M5 full-grid row-major ordering.
        n = len(row_indices)
        if n == self.n_rows * self.n_cols:
            rows_cpu = torch.as_tensor(row_indices, dtype=torch.long)
            cols_cpu = torch.as_tensor(col_indices, dtype=torch.long)

            expected_rows = torch.arange(
                self.n_rows, dtype=torch.long
            ).repeat_interleave(self.n_cols)
            expected_cols = torch.arange(
                self.n_cols, dtype=torch.long
            ).repeat(self.n_rows)

            if torch.equal(rows_cpu, expected_rows) and torch.equal(
                cols_cpu, expected_cols
            ):
                return self._full_grid_encoding.to(device=device)

        # General/subset path. Still fully tensorized; no Python coordinate
        # list-comprehensions or torch.arange() frequency construction.
        rows = torch.as_tensor(
            row_indices, dtype=torch.float32, device=device
        )
        cols = torch.as_tensor(
            col_indices, dtype=torch.float32, device=device
        )

        norm_rows = (
            rows / (self.n_rows - 1)
            if self.n_rows > 1
            else torch.zeros_like(rows)
        )
        norm_cols = (
            cols / (self.n_cols - 1)
            if self.n_cols > 1
            else torch.zeros_like(cols)
        )

        half = self.embedding_dim // 2
        row_enc = self._sinusoidal_1d_cached(
            norm_rows,
            half,
            self._row_div_term,
        )
        col_enc = self._sinusoidal_1d_cached(
            norm_cols,
            self.embedding_dim - half,
            self._col_div_term,
        )
        return torch.cat([row_enc, col_enc], dim=-1)
