"""
================================================================================
Unit Tests for M5 (Patch Generation)
================================================================================
Covers: patch count correctness, patch dimensions, reconstruction
correctness, padding correctness, metadata correctness, activity statistics
correctness, dataset integration, DataLoader integration, visualization,
report generation.

Run standalone:
    python test_m5_patches.py --dataset_root path/to/dataset
================================================================================
"""

import os
import argparse
import torch
import numpy as np

from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims, partition_voxel_grid, reconstruct_from_patches
from patch_statistics import compute_patch_activity_stats
from patch_adjacency import build_adjacency_map
from patch_generator import PatchGenerator, estimate_patch_generation_cost
from patch_visualization import visualize_patches
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m5_patch_collate import build_full_pipeline_dataloader


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


def run_all_m5_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M5 patch generation unit tests against: {dataset_root}\n")

    # Deliberately non-divisible dims to exercise padding
    C, H, W = 10, 50, 70
    voxel = torch.arange(C * H * W, dtype=torch.float32).reshape(C, H, W)

    # ---- 1. Patch count correctness (both padding modes) ----
    try:
        cfg_zero = PatchConfig(patch_size=16, padding_mode="zero")
        n_rows_z, n_cols_z, eff_h_z, eff_w_z = compute_patch_grid_dims(H, W, cfg_zero)
        expected_z = np.ceil(H / 16) * np.ceil(W / 16)
        patches_z, meta_z = partition_voxel_grid(voxel, cfg_zero)

        cfg_ign = PatchConfig(patch_size=16, padding_mode="ignore")
        n_rows_i, n_cols_i, eff_h_i, eff_w_i = compute_patch_grid_dims(H, W, cfg_ign)
        expected_i = (H // 16) * (W // 16)
        patches_i, meta_i = partition_voxel_grid(voxel, cfg_ign)

        ok = (len(meta_z) == expected_z and patches_z.shape[0] == expected_z
              and len(meta_i) == expected_i and patches_i.shape[0] == expected_i)
        tr.record("patch_count_correctness", ok,
                   f"zero: {len(meta_z)} (expected {int(expected_z)}), "
                   f"ignore: {len(meta_i)} (expected {int(expected_i)})")
    except Exception as e:
        tr.record("patch_count_correctness", False, str(e))

    # ---- 2. Patch dimensions ----
    try:
        ok = patches_z.shape[1:] == (C, 16, 16) and patches_i.shape[1:] == (C, 16, 16)
        tr.record("patch_dimensions", ok, f"zero shape={tuple(patches_z.shape)}, ignore shape={tuple(patches_i.shape)}")
    except Exception as e:
        tr.record("patch_dimensions", False, str(e))

    # ---- 3. Reconstruction correctness ----
    try:
        recon_z = reconstruct_from_patches(patches_z, meta_z, (C, eff_h_z, eff_w_z))
        matches_original_z = torch.equal(recon_z[:, :H, :W], voxel)
        padding_is_zero = torch.all(recon_z[:, H:, :] == 0) and torch.all(recon_z[:, :, W:] == 0)

        recon_i = reconstruct_from_patches(patches_i, meta_i, (C, eff_h_i, eff_w_i))
        matches_cropped_original_i = torch.equal(recon_i, voxel[:, :eff_h_i, :eff_w_i])

        ok = bool(matches_original_z and padding_is_zero and matches_cropped_original_i)
        tr.record("reconstruction_correctness", ok,
                   f"zero_matches={bool(matches_original_z)}, zero_padding_correct={bool(padding_is_zero)}, "
                   f"ignore_matches={bool(matches_cropped_original_i)}")
    except Exception as e:
        tr.record("reconstruction_correctness", False, str(e))

    # ---- 4. Padding correctness (dedicated: verify eff dims and that "ignore" < "zero") ----
    try:
        ok = (eff_h_z >= H and eff_w_z >= W and eff_h_i <= H and eff_w_i <= W
              and eff_h_z % 16 == 0 and eff_w_z % 16 == 0 and eff_h_i % 16 == 0 and eff_w_i % 16 == 0)
        tr.record("padding_correctness", ok,
                   f"zero eff=({eff_h_z},{eff_w_z}), ignore eff=({eff_h_i},{eff_w_i}), original=({H},{W})")
    except Exception as e:
        tr.record("padding_correctness", False, str(e))

    # ---- 5. Metadata correctness ----
    try:
        ok = True
        detail = ""
        for row in range(n_rows_z):
            for col in range(n_cols_z):
                idx = row * n_cols_z + col
                m = meta_z[idx]
                if not (m.patch_index == idx and m.row_index == row and m.col_index == col
                        and m.y0 == row * 16 and m.x0 == col * 16
                        and m.y1 == m.y0 + 16 and m.x1 == m.x0 + 16
                        and abs(m.center_y - (m.y0 + m.y1) / 2.0) < 1e-9
                        and m.temporal_dim == C and m.spatial_size == 16):
                    ok = False
                    detail = f"mismatch at row={row},col={col}: {m}"
                    break
            if not ok:
                break
        tr.record("metadata_correctness", ok, detail)
    except Exception as e:
        tr.record("metadata_correctness", False, str(e))

    # ---- 6. Activity statistics correctness (hand-crafted known values) ----
    try:
        cc, hh, ww = 10, 32, 32
        v = torch.zeros(cc, hh, ww)
        v[0, 0:16, 0:16] = 1.0
        v[1, 0:16, 16:32] = -2.0
        cfg = PatchConfig(patch_size=16, padding_mode="zero")
        vcfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        pp, mm = partition_voxel_grid(v, cfg)
        ss = compute_patch_activity_stats(pp, vcfg)
        ok = (abs(ss[0].total_event_count - 256.0) < 1e-6 and abs(ss[0].positive_event_count - 256.0) < 1e-6
              and abs(ss[1].total_event_count - 512.0) < 1e-6 and abs(ss[1].negative_event_count - 512.0) < 1e-6
              and abs(ss[0].event_density - 1.0) < 1e-6 and abs(ss[1].event_density - 2.0) < 1e-6)
        tr.record("activity_statistics_correctness", ok,
                   f"patch0={ss[0].total_event_count}/{ss[0].positive_event_count}, "
                   f"patch1={ss[1].total_event_count}/{ss[1].negative_event_count}")
    except Exception as e:
        tr.record("activity_statistics_correctness", False, str(e))

    # ---- 7. Adjacency correctness (corner has fewer neighbors than interior) ----
    try:
        adj4 = build_adjacency_map(4, 5, PatchConfig(adjacency_mode="4"))
        adj8 = build_adjacency_map(4, 5, PatchConfig(adjacency_mode="8"))
        corner_ok = len(adj4[0]) == 2 and len(adj8[0]) == 3
        interior_idx = 1 * 5 + 2  # row1,col2 -> interior for a 4x5 grid
        interior_ok = len(adj4[interior_idx]) == 4 and len(adj8[interior_idx]) == 8
        ok = corner_ok and interior_ok
        tr.record("adjacency_correctness", ok,
                   f"corner: 4-mode={len(adj4[0])}, 8-mode={len(adj8[0])}; "
                   f"interior: 4-mode={len(adj4[interior_idx])}, 8-mode={len(adj8[interior_idx])}")
    except Exception as e:
        tr.record("adjacency_correctness", False, str(e))

    # ---- 8. Dataset integration ----
    dataset = None
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        vcfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
        pcfg = PatchConfig(patch_size=16, padding_mode="zero")
        voxel_gen = VoxelGridGenerator(vcfg)
        gen = PatchGenerator(pcfg, vcfg)

        sample = dataset[0]
        v = voxel_gen.generate(sample)
        result = gen.generate(v)
        ok = result.patches.shape[0] == result.n_rows * result.n_cols == len(result.metadata) == len(result.stats)
        tr.record("dataset_integration", ok,
                   f"n_patches={result.patches.shape[0]}, grid=({result.n_rows},{result.n_cols})")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 9. DataLoader integration ----
    if dataset is not None:
        try:
            loader = build_full_pipeline_dataloader(dataset, dataset_root, batch_size=3, shuffle=False)
            batch = next(iter(loader))
            ok = (batch["patches"].ndim == 5 and batch["patches"].shape[0] == min(3, len(dataset))
                  and batch["patches"].shape[1] == batch["n_rows"] * batch["n_cols"])
            tr.record("dataloader_integration", ok, f"patches batch shape={tuple(batch['patches'].shape)}")
        except Exception as e:
            tr.record("dataloader_integration", False, str(e))
    else:
        tr.record("dataloader_integration", False, "skipped: no dataset")

    # ---- 10. Visualization ----
    if dataset is not None:
        try:
            vcfg = VoxelGridConfig(num_bins=10, polarity_encoding="signed")
            pcfg = PatchConfig(patch_size=16, padding_mode="zero")
            voxel_gen = VoxelGridGenerator(vcfg)
            gen = PatchGenerator(pcfg, vcfg)
            sample = dataset[0]
            v = voxel_gen.generate(sample)
            result = gen.generate(v)
            out_path = "test_m5_patch_viz.png"
            saved = visualize_patches(v, result.metadata, result.stats, vcfg, pcfg,
                                       result.n_rows, result.n_cols, out_path=out_path)
            ok = os.path.exists(saved) and os.path.getsize(saved) > 0
            tr.record("visualization", ok, f"saved to {saved}")
            if cleanup and ok:
                os.remove(saved)
        except Exception as e:
            tr.record("visualization", False, str(e))
    else:
        tr.record("visualization", False, "skipped: no dataset")

    # ---- 11. Report generation (importability smoke test) ----
    try:
        from generate_m5_report import generate_m5_report  # noqa
        tr.record("report_generation_importable", True)
    except Exception as e:
        tr.record("report_generation_importable", False, str(e))

    all_passed = tr.summary()
    return all_passed, tr.results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--window_us", type=int, default=200_000)
    args = ap.parse_args()

    passed, _ = run_all_m5_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
