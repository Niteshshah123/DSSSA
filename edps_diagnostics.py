"""
================================================================================
EDPS Collapse Diagnostic
================================================================================
Checks whether the learned importance score has effectively collapsed into
a disguised event-density threshold -- the specific failure mode flagged in
the design review. Computed automatically and surfaced in every M7 report,
not something that has to be remembered and checked by hand.

A high |correlation| is a WARNING, not a hard failure: some genuine
correlation between importance and density is expected and fine (dense
patches often ARE more important). The concern is specifically a
near-total collapse (|corr| above the configurable threshold), which would
mean the multi-cue scorer has learned nothing beyond what a one-line
density threshold already captures.
================================================================================
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np


def compute_score_density_correlation(scores: np.ndarray, densities: np.ndarray) -> float:
    if len(scores) < 2 or np.std(scores) == 0 or np.std(densities) == 0:
        return 0.0
    return float(np.corrcoef(scores, densities)[0, 1])


def run_collapse_diagnostic(scores: np.ndarray, densities: np.ndarray, warn_threshold: float) -> Dict[str, Any]:
    corr = compute_score_density_correlation(scores, densities)
    collapsed = abs(corr) > warn_threshold
    return {
        "score_density_correlation": corr,
        "warn_threshold": warn_threshold,
        "likely_collapsed_to_density_threshold": collapsed,
        "note": (
            "High |correlation| suggests the scorer may have degenerated into a "
            "disguised event-density threshold rather than using multi-cue "
            "information. This is expected to be uninformative before any "
            "detection-loss training has occurred (M7 has no training signal "
            "yet) -- re-check this diagnostic after M8-M10 training."
            if collapsed else
            "No evidence of collapse into a pure density threshold."
        ),
    }
