"""
================================================================================
EDPS Gradient Wiring Fix (M10)
================================================================================
PROBLEM (identified in the M10 design review): M8/M9's original collators
consumed EDPSOutput.selected_embeddings directly -- the raw, hard-selected
subset. Since hard selection (threshold > gate_threshold) is a non-
differentiable operation, detection-loss gradients could never reach EDPS's
scoring network through that path. The ONLY signal EDPS's scorer could ever
receive was the sparsity loss (which only shapes how MANY tokens are kept,
never WHICH ones) -- undermining the "learned importance" claim of the
whole project.

FIX: multiply each selected embedding by its own continuous soft gate value
before it's fed to the Transformer:

    gated = selected_embeddings * soft_gate[selected_indices].unsqueeze(-1)

This is applied IDENTICALLY in training and inference -- there is no
separate code path -- so it cannot introduce a train/inference mismatch.
Gate values for tokens that passed the threshold are typically > 0.5 (often
close to 1 once trained), so this is a mild, principled attenuation of the
forward pass, not a large distortion -- and now `d(detection_loss)/d(gate)`
is a well-defined, non-zero quantity, letting the scorer learn to rank
SELECTED tokens by whether keeping them actually helped.

KNOWN, DOCUMENTED LIMITATION: this wiring only reaches tokens EDPS decided
to keep. A token EDPS incorrectly dropped (a false negative -- e.g. an
actual object patch) never enters the detection loss at all, since it's
never in `selected_embeddings`. Only the sparsity loss (computed over ALL N
patches' scores, selected or not) touches such a token's gradient. This is
an inherent limitation of hard/discrete token selection generally (the
"credit assignment for discarded candidates" problem also faced by
DynamicViT-style methods), not specific to a coding error here. A fully
soft-masked alternative (scoring all N tokens densely, every epoch) would
fix it at the cost of the very sparsity benefit EDPS exists to provide, so
it is accepted as a documented trade-off rather than solved in M10 --
suitable material for the paper's limitations discussion.
================================================================================
"""

from __future__ import annotations

import torch

from edps_module import EDPSOutput


def compute_gated_selected_embeddings(edps_output: EDPSOutput) -> torch.Tensor:
    """
    Returns (K, D): selected_embeddings scaled by their own soft gate value.
    Used identically at train and inference time.
    """
    gate_values = edps_output.importance_scores[edps_output.binary_mask]  # (K,)
    if gate_values.shape[0] != edps_output.selected_embeddings.shape[0]:
        raise ValueError(
            f"Gate/embedding count mismatch: {gate_values.shape[0]} gate values "
            f"vs {edps_output.selected_embeddings.shape[0]} selected embeddings -- "
            f"binary_mask and selected_embeddings must be derived from the same "
            f"EDPSOutput and never re-ordered independently."
        )
    return edps_output.selected_embeddings * gate_values.unsqueeze(-1)
