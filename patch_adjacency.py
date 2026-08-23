from __future__ import annotations

from typing import Dict, List

from patch_config import PatchConfig


# Cache only topology-independent patch-grid dimensions. The adjacency values
# depend solely on (n_rows, n_cols, mode), so rebuilding the same graph for
# every window is unnecessary.
_ADJ_CACHE: Dict[tuple, Dict[int, List[int]]] = {}


def build_adjacency_map(
    n_rows: int, n_cols: int, config: PatchConfig
) -> Dict[int, List[int]]:
    key = (n_rows, n_cols, config.adjacency_mode)
    cached = _ADJ_CACHE.get(key)
    if cached is not None:
        # Return the cached immutable-in-practice lists as-is. Downstream M7
        # only reads them. This avoids rebuilding thousands of Python lists.
        return cached

    if config.adjacency_mode == "4":
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )

    adjacency: Dict[int, List[int]] = {}
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            neighbors = []
            for dr, dc in offsets:
                nr, nc = row + dr, col + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    neighbors.append(nr * n_cols + nc)
            adjacency[idx] = sorted(neighbors)

    _ADJ_CACHE[key] = adjacency
    return adjacency
