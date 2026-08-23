"""
================================================================================
M4 Preprocessing Configuration
================================================================================
Every threshold is exposed here (nothing hardcoded downstream) so the whole
preprocessing pipeline can be swept during the ablation study without
touching any filter's implementation code.

DESIGN PRINCIPLE (per approved M4 design discussion): defaults are
intentionally LENIENT. M4's job is to remove obvious sensor noise only --
EDPS, not M4, is responsible for deciding which active regions matter. A
lenient BAF/isolated-event threshold (e.g. min_neighbors=1, the loosest
non-trivial value) is the conservative choice; tightening these values is
an experiment to run later, not a default to ship.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    # ---- Master toggles (item 5: modular pipeline) ----
    use_baf: bool = True
    use_hot_pixel_filter: bool = True
    use_isolated_event_filter: bool = True
    use_normalization: bool = True

    # ---- Background Activity Filter (item 1) ----
    # For each event, count how many neighboring PIXELS within `baf_radius`
    # (Chebyshev distance) had their own most recent event within
    # `baf_time_window_us` of this event's timestamp. Keep the event iff
    # that count >= baf_min_neighbors.
    baf_radius: int = 1
    baf_time_window_us: int = 10_000
    baf_min_neighbors: int = 1  # deliberately not hardcoded stricter -- see module docstring

    # ---- Isolated Event Removal (item 2) ----
    # Same mechanism as BAF, but a SEPARATE, independently toggleable stage
    # with its own radius/window/threshold, applied to whatever survived BAF
    # (and hot-pixel filtering, per the requested pipeline order).
    iso_radius: int = 1
    iso_time_window_us: int = 5_000
    iso_min_neighbors: int = 1

    # ---- Hot Pixel / Hot Row Suppression (item 3) ----
    # Detected via statistical outlier detection over a FULL recording's
    # event-count distribution (never hardcoded coordinates). z-score
    # thresholds are deliberately high (conservative) by default -- only
    # flag pixels/rows that are extreme outliers, since suppression is a
    # blanket, whole-recording decision.
    hot_pixel_enable_pixel_mode: bool = True
    hot_pixel_enable_row_mode: bool = True
    hot_pixel_z_thresh: float = 8.0
    hot_row_z_thresh: float = 8.0
    hot_pixel_min_count: int = 50  # ignore near-empty pixels when computing z-scores

    # ---- Normalization (item 4) ----
    # NOTE: normalization produces ADDITIONAL fields (t_norm/x_norm/y_norm/
    # p_norm); it does NOT replace the integer (t, x, y) that M3's voxel
    # grid indexes with. Voxelization always consumes the filtered INTEGER
    # coordinates, never the normalized floats, so enabling/disabling
    # normalization cannot silently change M3's behavior.
    normalize_timestamp: bool = True
    normalize_coordinates: bool = True
    normalize_polarity: bool = False  # maps {0,1} -> {-1,+1} if enabled
