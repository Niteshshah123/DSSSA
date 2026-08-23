"""
================================================================================
Gate Score Logger (refinement 4)
================================================================================
Accumulates gate-score statistics across calls (batches, and later, training
epochs in M8+) so the evolution of gating behavior can be analyzed in the
paper -- e.g. "did the score distribution sharpen/polarize over training,
did the average keep-rate stabilize, etc."

Usage now (M7): logs scores produced during report generation, as a
demonstration/validation that the logging mechanism works end-to-end.
Usage later (M8-M10): the SAME class is reused inside the training loop,
called once per batch/epoch, with the accumulated history saved
periodically -- nothing about this class is specific to M7's untrained
weights.
================================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import torch


@dataclass
class GateScoreSnapshot:
    step: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    frac_above_threshold: float


class GateScoreLogger:
    def __init__(self):
        self.history: List[GateScoreSnapshot] = []
        self._all_scores: List[np.ndarray] = []   # kept for the final histogram, bounded by caller's usage

    def log(self, gate_scores: torch.Tensor, threshold: float, step: int = None):
        scores = gate_scores.detach().cpu().numpy().flatten()
        step = step if step is not None else len(self.history)
        snap = GateScoreSnapshot(
            step=step,
            mean=float(scores.mean()) if len(scores) else 0.0,
            std=float(scores.std()) if len(scores) else 0.0,
            min=float(scores.min()) if len(scores) else 0.0,
            max=float(scores.max()) if len(scores) else 0.0,
            median=float(np.median(scores)) if len(scores) else 0.0,
            frac_above_threshold=float((scores > threshold).mean()) if len(scores) else 0.0,
        )
        self.history.append(snap)
        self._all_scores.append(scores)

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"num_logged_steps": 0}
        means = [s.mean for s in self.history]
        keep_rates = [s.frac_above_threshold for s in self.history]
        return {
            "num_logged_steps": len(self.history),
            "overall_mean_score": float(np.mean(means)),
            "overall_std_score": float(np.std(means)),
            "overall_mean_keep_rate": float(np.mean(keep_rates)),
            "overall_std_keep_rate": float(np.std(keep_rates)),
        }

    def all_scores_flat(self) -> np.ndarray:
        if not self._all_scores:
            return np.array([])
        return np.concatenate(self._all_scores)

    def save(self, path: str):
        data = {
            "history": [s.__dict__ for s in self.history],
            "summary": self.summary(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def plot_histogram(self, out_path: str = "gate_score_history.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        all_scores = self.all_scores_flat()
        axes[0].hist(all_scores, bins=40, color="teal")
        axes[0].set_title(f"Gate score distribution\n(pooled over {len(self.history)} logged step(s))")
        axes[0].set_xlabel("gate score (post-sigmoid)")
        axes[0].set_ylabel("count")

        if len(self.history) > 1:
            steps = [s.step for s in self.history]
            means = [s.mean for s in self.history]
            keep_rates = [s.frac_above_threshold for s in self.history]
            ax2 = axes[1]
            ax2.plot(steps, means, label="mean gate score", color="teal")
            ax2.plot(steps, keep_rates, label="keep rate (frac > threshold)", color="darkorange")
            ax2.set_xlabel("logged step")
            ax2.legend()
            ax2.set_title("Gate score evolution")
        else:
            axes[1].text(0.5, 0.5, "Need >1 logged step\nto show evolution over time",
                          ha="center", va="center")
            axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close(fig)
        return out_path
