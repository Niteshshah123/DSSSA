"""
================================================================================
EDPS Sparsity Curriculum (M10)
================================================================================
Per the approved design (curriculum on EDPS's aggressiveness, not on which
components are frozen):

  Stage 1 (0 -> stage1_end_pct of training): lenient band
    [curriculum_start_min_ratio, curriculum_start_max_ratio] -- lets the
    Transformer/detection head learn real detection capability against a
    near-complete token set before EDPS becomes selective.

  Stage 2 (stage1_end_pct -> stage2_end_pct): linear anneal from the lenient
    band to the target band -- this is where EDPS actually learns to be
    selective, guided by both the sparsity loss AND (thanks to the M10
    gradient-wiring fix) real detection-loss gradients.

  Stage 3 (stage2_end_pct -> 1.0): band held fixed at the target
    [curriculum_target_min_ratio, curriculum_target_max_ratio] -- the
    Transformer/head fine-tune against the final, harder-pruned token
    distribution.

The forward path (EDPS -> Transformer -> head) is architecturally IDENTICAL
throughout -- only the sparsity loss's band bounds change -- so there is no
"EDPS-off" mode that's absent at deployment.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from training_config import TrainingConfig


@dataclass
class SparsityBand:
    min_ratio: float
    max_ratio: float
    stage: int  # 1, 2, or 3


def get_current_sparsity_band(progress_fraction: float, config: TrainingConfig) -> SparsityBand:
    """progress_fraction: current_epoch / total_epochs, in [0, 1]."""
    progress_fraction = max(0.0, min(1.0, progress_fraction))

    if progress_fraction < config.stage1_end_pct:
        return SparsityBand(
            min_ratio=config.curriculum_start_min_ratio,
            max_ratio=config.curriculum_start_max_ratio,
            stage=1,
        )

    if progress_fraction < config.stage2_end_pct:
        span = config.stage2_end_pct - config.stage1_end_pct
        t = (progress_fraction - config.stage1_end_pct) / span if span > 0 else 1.0
        min_ratio = config.curriculum_start_min_ratio + t * (config.curriculum_target_min_ratio - config.curriculum_start_min_ratio)
        max_ratio = config.curriculum_start_max_ratio + t * (config.curriculum_target_max_ratio - config.curriculum_start_max_ratio)
        return SparsityBand(min_ratio=min_ratio, max_ratio=max_ratio, stage=2)

    return SparsityBand(
        min_ratio=config.curriculum_target_min_ratio,
        max_ratio=config.curriculum_target_max_ratio,
        stage=3,
    )
