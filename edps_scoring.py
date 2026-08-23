"""
EDPS Scoring Network.

Behavior is unchanged. The expensive per-call Python construction of the
dense normalized adjacency matrix is retained as a compatibility fallback,
but EDPSModule now supports a prebuilt tensor adjacency.
"""
from __future__ import annotations
from typing import Dict,List,Optional
import torch
import torch.nn as nn
from edps_config import EDPSConfig

def build_normalized_adjacency_matrix(adjacency: Dict[int,List[int]], n_patches:int, device=None, dtype=torch.float32):
    A=torch.zeros((n_patches,n_patches),dtype=dtype,device=device)
    # Vectorize construction from metadata without changing edge weights.
    rows=[]; cols=[]
    for i,neighbors in adjacency.items():
        if not neighbors: continue
        rows.extend([i]*len(neighbors)); cols.extend(neighbors)
    if rows:
        r=torch.tensor(rows,device=A.device,dtype=torch.long)
        c=torch.tensor(cols,device=A.device,dtype=torch.long)
        deg=torch.bincount(r,minlength=n_patches).to(dtype=A.dtype)
        A.index_put_((r,c),1.0/deg[r],accumulate=False)
    return A

class EDPSScoringNetwork(nn.Module):
    def __init__(self,input_feature_dim:int,config:EDPSConfig):
        super().__init__()
        self.config=config
        self.local_proj=nn.Sequential(nn.Linear(input_feature_dim,config.local_feature_dim),nn.ReLU(inplace=True))
        combine_in_dim=config.local_feature_dim*(2 if config.use_neighbor_context else 1)
        self.combine=nn.Sequential(
            nn.Linear(combine_in_dim,config.scoring_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.scoring_hidden_dim,1)
        )
    def forward(self,features,adjacency_matrix=None):
        h=self.local_proj(features)
        if self.config.use_neighbor_context:
            if adjacency_matrix is None: raise ValueError("use_neighbor_context=True requires an adjacency_matrix.")
            neighbor_h=adjacency_matrix @ h
            combined=torch.cat([h,neighbor_h],dim=-1)
        else: combined=h
        return self.combine(combined).squeeze(-1)
