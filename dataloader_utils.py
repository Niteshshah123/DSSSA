"""
================================================================================
DataLoader Utilities (M2, items 5-6)
================================================================================
Event counts vary window-to-window, so raw (t, x, y, p) tensors CANNOT be
stacked into a single dense batch tensor the way fixed-size images can.
`collate_events` therefore returns a batch as a LIST of per-sample dicts
(each holding its own variable-length tensors) rather than attempting to
pad/stack them here.

WHY NOT PAD/STACK NOW: padding to a fixed max-event-count would be premature
at this stage of the pipeline -- voxelization (M3) is what converts variable-
length raw events into a FIXED-SIZE tensor per window (the voxel grid), which
*is* naturally stackable. Padding raw events now would just be thrown away
once voxelization is implemented, and would bias any "events per window"
statistics computed downstream. So M2's batches remain a list of variable-
length samples; M3's dataloader (or a wrapping collate_fn added at that
point) is where fixed-size stacking will actually happen.
================================================================================
"""

from __future__ import annotations

from typing import List, Dict, Any

from torch.utils.data import DataLoader

from gen1_dataset import EDPSGen1Dataset


def collate_events(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identity collate: keep the batch as a list of per-sample dicts."""
    return batch


def build_dataloader(
    dataset: EDPSGen1Dataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_events,
        **kwargs,
    )


def summarize_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick, human-readable summary of one batch -- useful for smoke-testing."""
    n_events = [len(s["t"]) for s in batch]
    n_boxes = [len(s["boxes"]) for s in batch]
    return {
        "batch_size": len(batch),
        "events_per_sample": n_events,
        "boxes_per_sample": n_boxes,
        "recordings": [s["recording_stem"] for s in batch],
        "window_ranges_us": [(s["t_start"], s["t_end"]) for s in batch],
    }
