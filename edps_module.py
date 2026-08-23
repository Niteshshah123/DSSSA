"""
EDPSModule - optimized orchestration.

Mathematics/selection behavior is unchanged. The optimization removes the
per-forward CPU construction and CPU->GPU transfer of the dense adjacency
matrix. A fixed grid adjacency can be cached after first construction.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List,Dict,Any
import torch
import torch.nn as nn
from edps_config import EDPSConfig
from edps_features import build_patch_features,feature_dim
from edps_scoring import EDPSScoringNetwork,build_normalized_adjacency_matrix
from patch_statistics import PatchActivityStats
from patch_metadata import PatchMetadata
from patch_embedding_module import EmbeddingMetadata

@dataclass
class EDPSOutput:
    selected_embeddings: torch.Tensor
    selected_metadata: List[PatchMetadata]
    binary_mask: torch.Tensor
    importance_scores: torch.Tensor
    soft_gated_embeddings: torch.Tensor
    token_reduction_stats: Dict[str,Any]

class EDPSModule(nn.Module):
    def __init__(self,config:EDPSConfig,embedding_dim:int,num_bins:int):
        super().__init__()
        self.config=config
        self.embedding_dim=embedding_dim
        self.scoring_net=EDPSScoringNetwork(feature_dim(embedding_dim,num_bins),config)
        self._adjacency_cache={}
    def _get_adjacency(self,adjacency,n,device,dtype):
        # Cache only by graph structure + device/dtype. This preserves exact
        # row-normalized semantics while avoiding rebuilding A every sample.
        key=(tuple(sorted((int(i),tuple(int(j) for j in js)) for i,js in adjacency.items())),n,str(device),dtype)
        cached=self._adjacency_cache.get(key)
        if cached is None:
            cached=build_normalized_adjacency_matrix(adjacency,n,device=device,dtype=dtype)
            self._adjacency_cache[key]=cached
        return cached
    def forward(self,embeddings,patch_stats,patch_metadata,embedding_metadata,adjacency):
        n=embeddings.shape[0]
        features=build_patch_features(embeddings,patch_stats,patch_metadata,embedding_metadata)
        adjacency_matrix=None
        if self.config.use_neighbor_context:
            adjacency_matrix=self._get_adjacency(adjacency,n,embeddings.device,embeddings.dtype)
        logits=self.scoring_net(features,adjacency_matrix)
        soft_gate=torch.sigmoid(logits)
        soft_gated_embeddings=embeddings*soft_gate.unsqueeze(-1)
        raw_gate_mask=soft_gate>self.config.gate_threshold
        if not self.training:
            gate_keep_rate=float(100*raw_gate_mask.float().mean().item())
            mean_gate=float(soft_gate.mean().item()); median_gate=float(soft_gate.median().item())
            min_gate=float(soft_gate.min().item()); max_gate=float(soft_gate.max().item())
        else: gate_keep_rate=50.0; mean_gate=median_gate=0.5; min_gate=0.0; max_gate=1.0
        if self.config.selection_mode=="baseline_no_pruning":
            hard_mask=torch.ones_like(raw_gate_mask,dtype=torch.bool); clamp_triggered=False; clamp_direction=None
        else:
            hard_mask,clamp_triggered,clamp_direction=self._apply_safety_clamp(raw_gate_mask.clone(),soft_gate,n)
        selected_idx=hard_mask.nonzero(as_tuple=True)[0]
        selected_embeddings=embeddings[selected_idx]
        selected_metadata=[patch_metadata[i] for i in selected_idx.tolist()]
        nr=int(hard_mask.sum().item()); removed=n-nr
        stats={"selection_mode":self.config.selection_mode,"gate_threshold":float(self.config.gate_threshold),
               "mean_gate":mean_gate,"median_gate":median_gate,"min_gate":min_gate,"max_gate":max_gate,
               "gate_keep_rate":gate_keep_rate,"n_input":n,"n_retained":nr,"n_removed":removed,
               "pct_retained":100*nr/n if n else 0.0,"pct_removed":100*removed/n if n else 0.0,
               "clamp_triggered":clamp_triggered,"clamp_direction":clamp_direction}
        return EDPSOutput(selected_embeddings,selected_metadata,hard_mask,soft_gate,soft_gated_embeddings,stats)
    def _apply_safety_clamp(self,hard_mask,soft_gate,n):
        min_count=max(1,round(self.config.clamp_min_ratio*n)); max_count=max(min_count,round(self.config.clamp_max_ratio*n))
        count=int(hard_mask.sum().item())
        if count<min_count:
            deficit=min_count-count; unselected=(~hard_mask).nonzero(as_tuple=True)[0]
            if len(unselected)>0:
                order=torch.argsort(soft_gate[unselected],descending=True); hard_mask=hard_mask.clone(); hard_mask[unselected[order[:deficit]]]=True
            return hard_mask,True,"min"
        if count>max_count:
            excess=count-max_count; selected=hard_mask.nonzero(as_tuple=True)[0]
            order=torch.argsort(soft_gate[selected]); hard_mask=hard_mask.clone(); hard_mask[selected[order[:excess]]]=False
            return hard_mask,True,"max"
        return hard_mask,False,None
