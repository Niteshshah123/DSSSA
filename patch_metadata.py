"""
================================================================================
Patch Metadata (M5, item 2)
================================================================================
Everything EDPS (M7) will need about a patch's position/identity, computed
once at partition time rather than recomputed later.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatchMetadata:
    patch_index: int         # flat, row-major index: patch_index = row_index * n_cols + col_index
    row_index: int
    col_index: int
    y0: int                  # pixel coordinates of the patch's top-left corner (in the padded/cropped grid)
    x0: int
    y1: int                  # exclusive bottom-right corner: patch occupies [y0:y1, x0:x1]
    x1: int
    center_y: float
    center_x: float
    temporal_dim: int        # number of channels/bins (C) in the source voxel grid
    spatial_size: int        # patch_size (patches are always square here)

    def to_dict(self):
        return self.__dict__.copy()
