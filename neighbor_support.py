"""
================================================================================
Neighbor Support Computation (shared primitive for BAF + Isolated-Event Removal)
================================================================================
Both the Background Activity Filter and Isolated-Event Removal ask the same
underlying question for each event: "how much recent activity happened near
me?" -- they just use different radius/window/threshold parameters and are
applied as separate, independently toggleable pipeline stages. Sharing this
primitive keeps the two filters' logic consistent and testable in one place,
without merging them into a single non-toggleable stage.

ALGORITHM (a practical, efficient approximation of classical Background
Activity Filtering, e.g. Delbruck 2008):
  Maintain a (height, width) "time surface" grid of each pixel's most recent
  event timestamp. Process events in ascending time order (already guaranteed
  by the validated M1 parser). For event i at (x, y, t):
    1. Look at every neighboring pixel within Chebyshev distance `radius`
       (excluding the center pixel itself).
    2. Count how many of those neighbors have a recorded last-event-time
       within `time_window_us` of t. Call this the "support count".
    3. The event is KEPT iff support_count >= min_neighbors.
    4. Update the time surface at (x, y) to t -- REGARDLESS of the keep/
       discard decision in step 3, because the "was there real activity
       nearby" question must be answered using the TRUE event history, not
       a history already thinned by this same filter. Filtering an event
       does not mean it didn't physically happen.

COMPLEXITY: O(N_events * K) time, where K = (2*radius+1)^2 - 1 neighbor
offsets (small constant, e.g. K=8 for radius=1). O(height * width) space
for the time-surface grid, independent of N_events.
================================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _neighbor_offsets(radius: int) -> List[Tuple[int, int]]:
    return [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if not (dy == 0 and dx == 0)
    ]


def compute_neighbor_support(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    radius: int,
    time_window_us: int,
) -> np.ndarray:
    """
    Returns support_count: int32 array of length len(t), where
    support_count[i] = number of distinct neighboring pixels (within
    Chebyshev `radius` of (x[i], y[i])) whose most recent PRIOR event
    occurred within `time_window_us` of t[i].

    REQUIRES: t must be sorted ascending (validated by the caller's use of
    EventParser-produced arrays; raises if violated, to fail loudly rather
    than silently produce wrong support counts).
    """
    n = len(t)
    support = np.zeros(n, dtype=np.int32)
    if n == 0:
        return support

    if np.any(np.diff(t) < 0):
        raise ValueError(
            "compute_neighbor_support requires timestamps sorted ascending; "
            "received non-monotonic input. This would silently corrupt the "
            "time-surface algorithm if not caught here."
        )

    offsets = _neighbor_offsets(radius)
    # -inf sentinel for "never seen" so the time_window_us comparison always fails
    last_time = np.full((height, width), -np.inf, dtype=np.float64)

    t = t.astype(np.float64)
    x = x.astype(np.int64)
    y = y.astype(np.int64)

    for i in range(n):
        ti, xi, yi = t[i], x[i], y[i]
        count = 0
        for dy, dx in offsets:
            ny, nx = yi + dy, xi + dx
            if 0 <= ny < height and 0 <= nx < width:
                lt = last_time[ny, nx]
                if (ti - lt) <= time_window_us:
                    count += 1
        support[i] = count
        last_time[yi, xi] = ti  # update AFTER checking -- see module docstring

    return support
