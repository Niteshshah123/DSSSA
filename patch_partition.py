from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from patch_config import PatchConfig
from patch_metadata import PatchMetadata


def compute_patch_grid_dims(
    height: int, width: int, config: PatchConfig
) -> Tuple[int, int, int, int]:
    ps = config.patch_size
    if config.padding_mode == "zero":
        n_rows = math.ceil(height / ps)
        n_cols = math.ceil(width / ps)
    else:
        n_rows = height // ps
        n_cols = width // ps
    return n_rows, n_cols, n_rows * ps, n_cols * ps


def _build_metadata(n_rows: int, n_cols: int, ps: int, c: int) -> List[PatchMetadata]:
    # Metadata is intentionally kept as Python objects because downstream M7
    # consumes PatchMetadata objects. Keep this loop out of tensor operations.
    metadata = []
    idx = 0
    for row in range(n_rows):
        y0 = row * ps
        y1 = y0 + ps
        center_y = (y0 + y1) / 2.0
        for col in range(n_cols):
            x0 = col * ps
            x1 = x0 + ps
            metadata.append(
                PatchMetadata(
                    patch_index=idx,
                    row_index=row,
                    col_index=col,
                    y0=y0,
                    x0=x0,
                    y1=y1,
                    x1=x1,
                    center_y=center_y,
                    center_x=(x0 + x1) / 2.0,
                    temporal_dim=c,
                    spatial_size=ps,
                )
            )
            idx += 1
    return metadata


def partition_voxel_grid(
    voxel: torch.Tensor, config: PatchConfig
) -> Tuple[torch.Tensor, List[PatchMetadata]]:
    """
    Partition (C,H,W) into row-major (N,C,P,P) patches.

    Important optimization:
    zero padding is created on the SAME device as the source voxel tensor.
    The previous implementation created it on CPU unconditionally.
    """
    if voxel.ndim != 3:
        raise ValueError(
            f"Expected voxel grid of shape (C, H, W), got shape {tuple(voxel.shape)}"
        )

    c, h, w = voxel.shape
    ps = config.patch_size
    n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(h, w, config)

    if n_rows == 0 or n_cols == 0:
        raise ValueError(
            f"patch_size={ps} is larger than the voxel grid ({h}x{w}) in "
            "'ignore' mode, producing zero patches."
        )

    if config.padding_mode == "zero":
        pad_h = eff_h - h
        pad_w = eff_w - w
        # F.pad avoids an explicit full-size CPU allocation and preserves
        # dtype/device. Padding order for (C,H,W): left,right,top,bottom.
        working = F.pad(voxel, (0, pad_w, 0, pad_h))
    else:
        working = voxel[:, :eff_h, :eff_w]

    patches_unfolded = working.unfold(1, ps, ps).unfold(2, ps, ps)
    patch_tensor = (
        patches_unfolded
        .permute(1, 2, 0, 3, 4)
        .reshape(-1, c, ps, ps)
    )

    metadata = _build_metadata(n_rows, n_cols, ps, c)
    return patch_tensor, metadata


def reconstruct_from_patches(
    patches: torch.Tensor,
    metadata: List[PatchMetadata],
    effective_shape: Tuple[int, int, int],
) -> torch.Tensor:
    c, eff_h, eff_w = effective_shape
    canvas = torch.zeros(
        (c, eff_h, eff_w),
        dtype=patches.dtype,
        device=patches.device,
    )
    for patch, meta in zip(patches, metadata):
        canvas[:, meta.y0:meta.y1, meta.x0:meta.x1] = patch
    return canvas
