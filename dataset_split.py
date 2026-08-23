"""
================================================================================
Dataset Splitter (M2, items 1-2)
================================================================================
Splits the dataset at the RECORDING level (never at the temporal-window
level) into train/val/test, per the approved SRS decision: "If a recording
belongs to the training set, all temporal windows generated from that
recording must also belong to the training set." This avoids the data
leakage that would occur if two windows cut from the same 60s recording
ended up in different splits.

The split is deterministic given a seed, and is persisted to a JSON manifest
so that: (a) re-running the pipeline reproduces the exact same split, and
(b) the manifest itself is a citable artifact for the paper's reproducibility
section.
================================================================================
"""

from __future__ import annotations

import os
import json
import glob
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class SplitManifest:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    train: List[str]
    val: List[str]
    test: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)

    def split_of(self, stem: str) -> Optional[str]:
        if stem in self.train:
            return "train"
        if stem in self.val:
            return "val"
        if stem in self.test:
            return "test"
        return None


def resolve_dataset_root(dataset_root: str) -> str:
    """
    If dataset_root does not contain recording pairs directly, automatically
    search subdirectories to find where event recordings (.dat) actually reside.
    """
    if not os.path.exists(dataset_root):
        return dataset_root

    # Check if dataset_root itself or pre-split subfolders contain recording pairs
    if len(list_recording_stems(dataset_root)) > 0:
        return dataset_root
    for folder in ["train", "val", "detection_dataset", "gen1"]:
        sub = os.path.join(dataset_root, folder)
        if os.path.isdir(sub) and len(list_recording_stems(sub)) > 0:
            if folder in ["train", "val"]:
                return dataset_root
            return sub

    # Walk subdirectories to find where *.dat files live
    for root, dirs, files in os.walk(dataset_root):
        if any(f.endswith(".dat") for f in files):
            if len(list_recording_stems(root)) > 0:
                return root
            parent = os.path.dirname(root)
            if len(list_recording_stems(parent)) > 0:
                return parent

    return dataset_root


def list_recording_stems(dataset_root: str) -> List[str]:
    """Finds every event recording file with a matching bbox annotation pair."""
    if not os.path.exists(dataset_root):
        return []

    dat_files = sorted(glob.glob(os.path.join(dataset_root, "*_td.dat")))
    if not dat_files:
        dat_files = sorted(glob.glob(os.path.join(dataset_root, "**", "*_td.dat"), recursive=True))
    if not dat_files:
        dat_files = sorted(glob.glob(os.path.join(dataset_root, "**", "*.dat"), recursive=True))

    stems = []
    for f in dat_files:
        rel_path = os.path.relpath(f, dataset_root)
        if rel_path.endswith("_td.dat"):
            stem_rel = rel_path[:-len("_td.dat")].replace("\\", "/")
        elif rel_path.endswith(".dat"):
            stem_rel = rel_path[:-len(".dat")].replace("\\", "/")
        else:
            continue

        dir_name = os.path.dirname(f)
        base_name = os.path.basename(stem_rel)

        bbox1 = os.path.join(dataset_root, stem_rel + "_bbox.npy")
        bbox2 = os.path.join(dataset_root, stem_rel + ".npy")
        npy_in_dir = glob.glob(os.path.join(dir_name, base_name + "*npy"))

        if os.path.exists(bbox1) or os.path.exists(bbox2) or len(npy_in_dir) > 0:
            stems.append(stem_rel)

    return sorted(list(set(stems)))


def create_split(
    dataset_root: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> SplitManifest:
    """
    Recording-level split. Shuffles the list of recording stems with a fixed
    seed, then partitions by count according to the given ratios. Every
    window later generated from a given recording inherits that recording's
    split assignment.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    resolved_root = resolve_dataset_root(dataset_root)

    # Check if dataset is pre-partitioned into train/val subfolders on disk
    train_dir = os.path.join(resolved_root, "train")
    val_dir = os.path.join(resolved_root, "val")
    test_dir = os.path.join(resolved_root, "test")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        train_stems = ["train/" + s for s in list_recording_stems(train_dir)]
        val_stems = ["val/" + s for s in list_recording_stems(val_dir)]
        test_stems = ["test/" + s for s in list_recording_stems(test_dir)] if os.path.isdir(test_dir) else []
        if len(train_stems) > 0 and len(val_stems) > 0:
            total_n = len(train_stems) + len(val_stems) + len(test_stems)
            manifest = SplitManifest(
                seed=seed,
                train_ratio=len(train_stems) / total_n,
                val_ratio=len(val_stems) / total_n,
                test_ratio=len(test_stems) / total_n,
                train=train_stems, val=val_stems, test=test_stems,
            )
            return manifest

    stems = list_recording_stems(resolved_root)
    if len(stems) == 0:
        found_subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))] if os.path.exists(dataset_root) else []
        raise ValueError(
            f"No matched *.dat/*_bbox.npy pairs found in dataset_root='{dataset_root}' "
            f"(resolved path: '{resolved_root}').\n"
            f"Subdirectories found inside dataset_root: {found_subdirs}.\n"
            f"Please check your Google Drive folder structure and pass the directory "
            f"that directly contains the dataset files or subfolders."
        )

    rng = random.Random(seed)
    shuffled = stems[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # test gets the remainder, so rounding never drops/duplicates a recording
    n_test = n - n_train - n_val

    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])

    manifest = SplitManifest(
        seed=seed, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        train=train, val=val, test=test,
    )
    _validate_manifest(manifest, stems)
    return manifest


def _validate_manifest(manifest: SplitManifest, all_stems: List[str]):
    all_assigned = manifest.train + manifest.val + manifest.test
    if len(all_assigned) != len(all_stems):
        raise AssertionError(
            f"Split manifest assigns {len(all_assigned)} recordings but "
            f"{len(all_stems)} were found -- a recording was lost or duplicated."
        )
    if len(set(all_assigned)) != len(all_assigned):
        raise AssertionError("Split manifest contains a duplicate recording across splits.")
    if set(all_assigned) != set(all_stems):
        raise AssertionError("Split manifest does not cover exactly the discovered recordings.")


def save_manifest(manifest: SplitManifest, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(path: str) -> SplitManifest:
    with open(path) as f:
        d = json.load(f)
    return SplitManifest(**d)
