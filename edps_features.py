"""
EDPS Feature Construction
GPU/vectorized implementation.

The feature definition is unchanged:
[embedding, log1p(total), log1p(pos), log1p(neg), density,
 polarity_ratio, normalized temporal profile, norm_row, norm_col]

The previous implementation built a NumPy matrix and filled it with nested
Python loops for every patch, then copied it CPU -> GPU. This version creates
the same derived features as torch tensors and performs one device transfer
for metadata-derived values.
"""
from __future__ import annotations
from typing import List
import math
import torch
from patch_statistics import PatchActivityStats
from patch_metadata import PatchMetadata
from patch_embedding_module import EmbeddingMetadata

EPS=1e-6

def build_patch_features(embeddings, patch_stats, patch_metadata, embedding_metadata):
    n=embeddings.shape[0]
    if not (len(patch_stats)==len(patch_metadata)==len(embedding_metadata)==n):
        raise ValueError(
            f"Length mismatch: embeddings={n}, patch_stats={len(patch_stats)}, "
            f"patch_metadata={len(patch_metadata)}, embedding_metadata={len(embedding_metadata)} "
            "-- all must correspond 1:1 by patch index."
        )
    if n==0:
        raise ValueError("Cannot build EDPS features for zero patches.")
    num_bins=len(patch_stats[0].temporal_activity_distribution)
    device=embeddings.device
    dtype=embeddings.dtype

    # Metadata is CPU/Python by design. Construct one compact tensor and move it
    # once, instead of NumPy allocation + per-patch loops + CPU->GPU transfer.
    rows=torch.tensor([float(m.norm_row) for m in embedding_metadata], device=device, dtype=dtype)
    cols=torch.tensor([float(m.norm_col) for m in embedding_metadata], device=device, dtype=dtype)
    counts=torch.tensor(
        [[max(0.0,float(s.total_event_count)),
          max(0.0,float(s.positive_event_count)),
          max(0.0,float(s.negative_event_count)),
          float(s.event_density)]
         for s in patch_stats], device=device, dtype=dtype)

    log_counts=torch.log1p(counts[:, :3])
    pos=counts[:,1]; neg=counts[:,2]
    pol=pos/(pos+neg+EPS)

    temporal=torch.tensor(
        [list(s.temporal_activity_distribution) for s in patch_stats],
        device=device, dtype=dtype
    )
    td_abs=temporal.abs().sum(dim=1, keepdim=True)
    temporal=torch.where(td_abs > 0, temporal/td_abs.clamp_min(EPS), torch.zeros_like(temporal))

    derived=torch.cat([log_counts, counts[:,3:4], pol.unsqueeze(1), temporal, rows.unsqueeze(1), cols.unsqueeze(1)], dim=1)
    return torch.cat([embeddings, derived], dim=1)

def feature_dim(embedding_dim:int,num_bins:int)->int:
    return embedding_dim+5+num_bins+2
