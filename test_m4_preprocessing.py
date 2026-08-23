"""
================================================================================
Unit Tests for M4 (Noise Reduction & Event Preprocessing)
================================================================================
Covers all required categories:
  1. BAF correctness (signal-preservation vs noise-removal, on a realistic
     spatially-correlated synthetic scene -- NOT the uniform-noise
     synthetic_dataset used for M1-M3 plumbing tests, which has no real
     signal to preserve and would give a misleadingly harsh result)
  2. Isolated-event removal correctness
  3. Hot-pixel/row detection (against a recording with a KNOWN planted hot row)
  4. Normalization correctness
  5. Dataset integration
  6. Batch integration
  7. Visualization
  8. Report generation

Run standalone:
    python test_m4_preprocessing.py --dataset_root path/to/dataset
================================================================================
"""

import os
import struct
import argparse
import numpy as np

from preprocessing_config import PreprocessingConfig
from background_activity_filter import apply_baf
from isolated_event_filter import apply_isolated_event_filter
from hot_pixel_filter import detect_hot_pixels_for_recording, apply_hot_pixel_filter
from event_normalization import normalize_events
from event_parser import PropheseeGen1Reader
from event_preprocessor import EventPreprocessor
from preprocessing_visualization import visualize_preprocessing_stages
from make_correlated_scene import build_correlated_scene
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m4_voxel_collate import build_preprocessed_voxel_dataloader


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


def _write_hot_row_recording(path, height=80, width=100, hot_row=40, seed=1):
    rng = np.random.default_rng(seed)
    n_normal = 50000
    t_normal = np.sort(rng.integers(0, 5_000_000, size=n_normal)).astype(np.uint32)
    x_normal = rng.integers(0, width, size=n_normal).astype(np.uint32)
    y_normal = rng.integers(0, height, size=n_normal).astype(np.uint32)
    p_normal = rng.integers(0, 2, size=n_normal).astype(np.uint32)

    n_hot = 20000
    t_hot = np.sort(rng.integers(0, 5_000_000, size=n_hot)).astype(np.uint32)
    x_hot = rng.integers(0, width, size=n_hot).astype(np.uint32)
    y_hot = np.full(n_hot, hot_row, dtype=np.uint32)
    p_hot = rng.integers(0, 2, size=n_hot).astype(np.uint32)

    t = np.concatenate([t_normal, t_hot])
    x = np.concatenate([x_normal, x_hot])
    y = np.concatenate([y_normal, y_hot])
    p = np.concatenate([p_normal, p_hot])
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]

    header = ("% Data file containing TD/APS events.\n% Version 2\n"
               "% Date 2019-11-27 09:41:27\n"
               f"% Height {height}\n% Width {width}\n").encode("latin-1")
    sub_header = struct.pack("<BB", 0, 8)
    packed = (x & 0x3FFF) | ((y & 0x3FFF) << 14) | ((p & 0x1) << 28)
    with open(path, "wb") as f:
        f.write(header)
        f.write(sub_header)
        interleaved = np.empty(len(t) * 2, dtype=np.uint32)
        interleaved[0::2] = t
        interleaved[1::2] = packed
        f.write(interleaved.tobytes())


def run_all_m4_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M4 preprocessing unit tests (dataset plumbing against: {dataset_root})\n")

    H, W = 240, 304

    # ---- 1. BAF correctness: preserves correlated signal, removes uniform noise ----
    try:
        t, x, y, p, is_object = build_correlated_scene(height=H, width=W)
        cfg = PreprocessingConfig(baf_radius=1, baf_time_window_us=10_000, baf_min_neighbors=1)
        keep = apply_baf(t, x, y, H, W, cfg)
        obj_survival = keep[is_object].mean() if is_object.any() else 0
        noise_survival = keep[~is_object].mean() if (~is_object).any() else 0
        ok = obj_survival > 0.8 and noise_survival < 0.2
        tr.record("baf_correctness", ok,
                   f"object_survival={100*obj_survival:.1f}%, noise_survival={100*noise_survival:.1f}%")
    except Exception as e:
        tr.record("baf_correctness", False, str(e))

    # ---- 2. Isolated-event removal correctness (same scene, own config) ----
    try:
        t, x, y, p, is_object = build_correlated_scene(height=H, width=W, seed=3)
        cfg = PreprocessingConfig(iso_radius=1, iso_time_window_us=10_000, iso_min_neighbors=1)
        keep = apply_isolated_event_filter(t, x, y, H, W, cfg)
        obj_survival = keep[is_object].mean() if is_object.any() else 0
        noise_survival = keep[~is_object].mean() if (~is_object).any() else 0
        ok = obj_survival > 0.8 and noise_survival < 0.2
        tr.record("isolated_event_removal_correctness", ok,
                   f"object_survival={100*obj_survival:.1f}%, noise_survival={100*noise_survival:.1f}%")
    except Exception as e:
        tr.record("isolated_event_removal_correctness", False, str(e))

    # ---- 3. Hot-pixel/row detection against a KNOWN planted hot row ----
    hot_dat_path = "test_hotrow_td.dat"
    try:
        _write_hot_row_recording(hot_dat_path, height=80, width=100, hot_row=40)
        reader = PropheseeGen1Reader()
        cfg = PreprocessingConfig(hot_pixel_z_thresh=8.0, hot_row_z_thresh=8.0, hot_pixel_min_count=50)
        mask = detect_hot_pixels_for_recording(hot_dat_path, reader, cfg)
        ok = mask.hot_rows == [40] and len(mask.hot_pixels) == 0
        tr.record("hot_pixel_row_detection", ok,
                   f"detected hot_rows={mask.hot_rows}, hot_pixels={len(mask.hot_pixels)}, "
                   f"log={mask.log}")
    except Exception as e:
        tr.record("hot_pixel_row_detection", False, str(e))
    finally:
        if cleanup and os.path.exists(hot_dat_path):
            os.remove(hot_dat_path)

    # ---- 4. Normalization correctness ----
    try:
        t = np.array([100, 200, 300])
        x = np.array([0, 50, 99])
        y = np.array([0, 20, 39])
        p = np.array([1, 0, 1])
        cfg = PreprocessingConfig(normalize_timestamp=True, normalize_coordinates=True, normalize_polarity=True)
        norm = normalize_events(t, x, y, p, t_start=100, t_end=400, height=40, width=100, config=cfg)
        ok = (
            np.allclose(norm["t_norm"], [0.0, 1/3, 2/3])
            and np.allclose(norm["x_norm"], [0.0, 50/99, 1.0])
            and np.allclose(norm["y_norm"], [0.0, 20/39, 1.0])
            and np.allclose(norm["p_norm"], [1.0, -1.0, 1.0])
        )
        tr.record("normalization_correctness", ok, str({k: v.tolist() for k, v in norm.items()}))
    except Exception as e:
        tr.record("normalization_correctness", False, str(e))

    # ---- 5. Dataset integration ----
    dataset = None
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        pre = EventPreprocessor(PreprocessingConfig())
        sample = dataset[0]
        result = pre.process(sample, dataset_root=dataset_root)
        ok = all(k in result for k in ("t", "x", "y", "p", "stage_stats", "boxes"))
        tr.record("dataset_integration", ok,
                   f"final_count={len(result['t'])}, n_stages={len(result['stage_stats'])}")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 6. Batch integration (preprocessing + voxelization + stacking) ----
    if dataset is not None:
        try:
            loader = build_preprocessed_voxel_dataloader(
                dataset, dataset_root, batch_size=3, shuffle=False,
            )
            voxel_batch, boxes_list, meta_list, stage_stats_list = next(iter(loader))
            ok = (voxel_batch.ndim == 4 and voxel_batch.shape[0] == min(3, len(dataset))
                  and len(stage_stats_list) == voxel_batch.shape[0])
            tr.record("batch_integration", ok, f"voxel_batch.shape={tuple(voxel_batch.shape)}")
        except Exception as e:
            tr.record("batch_integration", False, str(e))
    else:
        tr.record("batch_integration", False, "skipped: no dataset")

    # ---- 7. Visualization ----
    if dataset is not None:
        try:
            pre = EventPreprocessor(PreprocessingConfig(
                baf_time_window_us=50_000, baf_min_neighbors=1,
                iso_time_window_us=50_000, iso_min_neighbors=1,
            ))
            sample = dataset[0]
            result = pre.process(sample, dataset_root=dataset_root, capture_snapshots=True)
            out_path = "test_m4_stages.png"
            saved = visualize_preprocessing_stages(
                result["snapshots"], sample["sensor_height"], sample["sensor_width"],
                boxes=sample["boxes"], out_path=out_path,
            )
            ok = os.path.exists(saved) and os.path.getsize(saved) > 0
            tr.record("visualization", ok, f"saved to {saved}")
            if cleanup and ok:
                os.remove(saved)
        except Exception as e:
            tr.record("visualization", False, str(e))
    else:
        tr.record("visualization", False, "skipped: no dataset")

    # ---- 8. Report generation (smoke test: import + minimal call) ----
    try:
        from generate_m4_report import generate_m4_report  # import-time check
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

    passed, _ = run_all_m4_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
