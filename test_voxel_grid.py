"""
================================================================================
Unit Tests for M3 (Voxel Grid Generation)
================================================================================
Run standalone:
    python test_voxel_grid.py --dataset_root path/to/dataset
================================================================================
"""

import os
import argparse
import numpy as np
import torch

from voxel_grid import VoxelGridConfig, events_to_voxel_grid, VoxelGridGenerator, estimate_voxelization_cost
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from voxel_collate import build_voxel_dataloader
from voxel_visualization import visualize_voxel_grid


class TestResult:
    def __init__(self):
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))

    def summary(self):
        n_pass = sum(1 for _, p, _ in self.results if p)
        n_total = len(self.results)
        print(f"\n{n_pass}/{n_total} tests passed.")
        return n_pass == n_total


def run_all_voxel_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M3 voxel grid unit tests against: {dataset_root}\n")

    H, W = 20, 20

    # ---- 1. Shape correctness: signed vs. separate_channels ----
    try:
        t = np.array([0, 500, 999])
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        p = np.array([1, 0, 1])

        cfg_signed = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        g1 = events_to_voxel_grid(t, x, y, p, 0, 1000, H, W, cfg_signed)
        ok1 = g1.shape == (10, H, W)

        cfg_sep = VoxelGridConfig(num_bins=10, polarity_encoding="separate_channels")
        g2 = events_to_voxel_grid(t, x, y, p, 0, 1000, H, W, cfg_sep)
        ok2 = g2.shape == (20, H, W)

        tr.record("shape_correctness", ok1 and ok2,
                  f"signed shape={g1.shape}, separate shape={g2.shape}")
    except Exception as e:
        tr.record("shape_correctness", False, str(e))

    # ---- 2. Conservation invariants (both modes) ----
    # NOTE: for "signed" encoding, ON and OFF events landing in the same
    # (bin, y, x) cell legitimately cancel -- that is what "signed net
    # polarity" means. So sum(|grid|) == n_events is NOT a valid invariant
    # under realistic data with spatial collisions. The exact invariant
    # that DOES hold regardless of collisions is: sum(grid) == n_ON - n_OFF,
    # since each event's weight (w0 + w1) always sums to 1 wherever it lands.
    # For "separate_channels" (non-negative, no cancellation possible),
    # sum(grid) == n_events exactly.
    try:
        n = 5000
        rng = np.random.default_rng(0)
        t = np.sort(rng.integers(0, 100000, size=n))
        x = rng.integers(0, W, size=n)
        y = rng.integers(0, H, size=n)
        p = rng.integers(0, 2, size=n)
        n_on = int((p > 0).sum())
        n_off = n - n_on

        for mode in ("bilinear", "count"):
            cfg = VoxelGridConfig(num_bins=10, temporal_mode=mode, polarity_encoding="signed")
            g = events_to_voxel_grid(t, x, y, p, 0, 100000, H, W, cfg)
            net = g.sum()
            expected_net = n_on - n_off
            ok = abs(net - expected_net) < 1e-3
            tr.record(f"conservation_signed_net_polarity_{mode}", ok,
                       f"sum(grid)={net:.3f}, expected net polarity {expected_net}")

            cfg2 = VoxelGridConfig(num_bins=10, temporal_mode=mode, polarity_encoding="separate_channels")
            g2 = events_to_voxel_grid(t, x, y, p, 0, 100000, H, W, cfg2)
            total2 = g2.sum()
            ok2 = abs(total2 - n) < 1e-3
            tr.record(f"conservation_separate_total_count_{mode}", ok2, f"sum(grid)={total2}, expected {n}")
    except Exception as e:
        tr.record("conservation", False, str(e))

    # ---- 3. Boundary conditions: events at t_start and t_end-1 ----
    try:
        cfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        t_edge = np.array([0, 999])
        x_edge = np.array([0, 0])
        y_edge = np.array([0, 0])
        p_edge = np.array([1, 1])
        g = events_to_voxel_grid(t_edge, x_edge, y_edge, p_edge, 0, 1000, H, W, cfg)
        # first event should land (at least partially) in bin 0, last event near bin num_bins-1
        ok = g[0, 0, 0] > 0 and g[-1, 0, 0] > 0
        tr.record("boundary_conditions", ok, f"bin0={g[0,0,0]:.3f}, last_bin={g[-1,0,0]:.3f}")
    except Exception as e:
        tr.record("boundary_conditions", False, str(e))

    # ---- 4. Empty-event window does not crash, returns zero grid ----
    try:
        cfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        g = events_to_voxel_grid(np.array([]), np.array([]), np.array([]), np.array([]), 0, 1000, H, W, cfg)
        ok = g.shape == (10, H, W) and g.sum() == 0
        tr.record("empty_window_handling", ok)
    except Exception as e:
        tr.record("empty_window_handling", False, str(e))

    # ---- 5. Integration with real Dataset sample (VoxelGridGenerator) ----
    dataset = None
    sample = None
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        sample = dataset[0]
        gen = VoxelGridGenerator(VoxelGridConfig(num_bins=10, polarity_encoding="signed"))
        voxel = gen.generate(sample)
        ok = (voxel.shape == (10, sample["sensor_height"], sample["sensor_width"])
              and torch.is_tensor(voxel))
        tr.record("dataset_integration", ok, f"voxel.shape={tuple(voxel.shape)}")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 6. Batch stacking via VoxelizedBatchCollator ----
    voxel_batch = None
    boxes_list = None
    if dataset is not None:
        try:
            loader = build_voxel_dataloader(
                dataset, voxel_config=VoxelGridConfig(num_bins=10), batch_size=3, shuffle=False
            )
            voxel_batch, boxes_list, meta_list = next(iter(loader))
            ok = (voxel_batch.ndim == 4 and voxel_batch.shape[0] == min(3, len(dataset))
                  and len(boxes_list) == voxel_batch.shape[0] and len(meta_list) == voxel_batch.shape[0])
            tr.record("batch_stacking", ok, f"voxel_batch.shape={tuple(voxel_batch.shape)}")
        except Exception as e:
            tr.record("batch_stacking", False, str(e))
    else:
        tr.record("batch_stacking", False, "skipped: no dataset")

    # ---- 7. Visualization ----
    if dataset is not None and sample is not None:
        try:
            gen = VoxelGridGenerator(VoxelGridConfig(num_bins=10, polarity_encoding="signed"))
            voxel = gen.generate(sample)
            out_path = "test_voxel_visualization.png"
            result_path = visualize_voxel_grid(voxel, gen.config, out_path=out_path, boxes=sample["boxes"])
            ok = os.path.exists(result_path) and os.path.getsize(result_path) > 0
            tr.record("visualization", ok, f"saved to {result_path}")
            if cleanup and ok:
                os.remove(result_path)
        except Exception as e:
            tr.record("visualization", False, str(e))
    else:
        tr.record("visualization", False, "skipped: no dataset/sample")

    # ---- 8. Cost estimator sanity ----
    try:
        cfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        cost = estimate_voxelization_cost(10000, cfg, 240, 304)
        ok = cost["estimated_flops"] > 0 and cost["output_shape"] == (10, 240, 304)
        tr.record("cost_estimator", ok, str(cost))
    except Exception as e:
        tr.record("cost_estimator", False, str(e))

    all_passed = tr.summary()
    return all_passed, tr.results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--window_us", type=int, default=200_000)
    args = ap.parse_args()

    passed, _ = run_all_voxel_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
