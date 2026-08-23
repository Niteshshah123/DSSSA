"""
================================================================================
M6 Research Log Report Generator
================================================================================
Runs the M6 unit test suite, computes embedding statistics over sampled
windows, the computational complexity analysis (params/FLOPs/latency/
memory), saves an example embedding visualization, and writes
M6_report.json / M6_report.md.

Usage:
    python generate_m6_report.py --dataset_root /path/to/dataset --output_dir /content/m6_report
================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import torch

from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from event_preprocessor import EventPreprocessor
from preprocessing_config import PreprocessingConfig
from voxel_grid import VoxelGridConfig, VoxelGridGenerator
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims
from patch_generator import PatchGenerator
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule, count_parameters
from embedding_visualization import visualize_embeddings
from embedding_cost_analysis import measure_embedding_module_cost
from test_m6_embedding import run_all_m6_tests


def generate_m6_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    patch_size: int = 16,
    embedding_dim: int = 256,
    projection_type: str = "linear",
    num_sample_windows: int = 20,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_m6_tests(dataset_root, window_us=window_us)

    voxel_config = VoxelGridConfig()
    patch_config = PatchConfig(patch_size=patch_size)
    embedding_config = PatchEmbeddingConfig(embedding_dim=embedding_dim, projection_type=projection_type)

    manifest = create_split(dataset_root, seed=seed)
    dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)

    preprocessor = EventPreprocessor(PreprocessingConfig())
    voxel_gen = VoxelGridGenerator(voxel_config)
    patch_gen = PatchGenerator(patch_config, voxel_config)

    n_rows, n_cols, _, _ = compute_patch_grid_dims(
        dataset[0]["sensor_height"], dataset[0]["sensor_width"], patch_config
    )
    embedding_module = PatchEmbeddingModule(
        embedding_config, in_channels=voxel_config.num_channels, patch_size=patch_config.patch_size,
        n_rows=n_rows, n_cols=n_cols, polarity_encoding=voxel_config.polarity_encoding,
    )

    n_sample = min(num_sample_windows, len(dataset))
    embedding_norms = []
    example_embeddings = None
    example_patch_result = None
    example_patches = None
    example_metadata = None

    for i in range(n_sample):
        sample = dataset[i]
        cleaned = preprocessor.process(sample, dataset_root=dataset_root)
        voxel = voxel_gen.generate({
            "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
            "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
            "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
        })
        patch_result = patch_gen.generate(voxel)
        with torch.no_grad():
            embeddings, meta_out = embedding_module(patch_result.patches, patch_result.metadata)

        embedding_norms.extend(embeddings.norm(dim=1).tolist())

        if example_embeddings is None:
            example_embeddings = embeddings
            example_patch_result = patch_result
            example_patches = patch_result.patches
            example_metadata = patch_result.metadata

    viz_path = os.path.join(output_dir, "embedding_visualization_example.png")
    activity = np.array([s.total_event_count for s in example_patch_result.stats])
    visualize_embeddings(example_embeddings, out_path=viz_path, activity=activity, include_tsne=True)

    cost = measure_embedding_module_cost(embedding_module, example_patches, example_metadata)

    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M6 - Patch Embedding",
        "config": {
            "embedding_dim": embedding_config.embedding_dim,
            "projection_type": embedding_config.projection_type,
            "positional_encoding_type": embedding_config.positional_encoding_type,
            "use_temporal_encoding": embedding_config.use_temporal_encoding,
            "temporal_encoding_type": embedding_config.temporal_encoding_type,
            "patch_size": patch_config.patch_size,
            "n_rows": n_rows, "n_cols": n_cols,
        },
        "embedding_statistics": {
            "num_windows_sampled": n_sample,
            "num_patches_per_window": n_rows * n_cols,
            "avg_embedding_norm": float(np.mean(embedding_norms)) if embedding_norms else 0.0,
            "std_embedding_norm": float(np.std(embedding_norms)) if embedding_norms else 0.0,
        },
        "computational_complexity": cost,
        "parameter_count": count_parameters(embedding_module),
        "example_visualization": viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
    }

    json_path = os.path.join(output_dir, "M6_report.json")
    md_path = os.path.join(output_dir, "M6_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M6 - Patch Embedding Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Embedding Statistics (aggregated over {n_sample} sampled windows)\n\n")
        for k, v in report["embedding_statistics"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Parameter Count\n\n- **total_parameters**: {report['parameter_count']}\n")

        f.write("\n## Computational Complexity Analysis\n\n")
        for k, v in cost.items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Example Visualization\n\n- `{viz_path}`\n")

        f.write(
            "\n_Research requirement check: M6 generates an embedding for every input patch and "
            "performs no ranking, pruning, or importance scoring -- confirmed by the "
            "'no_pruning_every_patch_embedded' unit test below, which checks that output count "
            "always equals input patch count._\n"
        )

        f.write("\n## Unit Test Results\n\n")
        for t in report["unit_test_results"]:
            status = "PASS" if t["passed"] else "FAIL"
            detail = f" — {t['detail']}" if t["detail"] else ""
            f.write(f"- [{status}] {t['name']}{detail}\n")

        f.write(f"\n**All tests passed: {report['all_unit_tests_passed']}**\n")
        f.write(f"\nExecution time: {report['execution_time_seconds']:.2f}s\n")

    print(f"\nReport written to:\n  {json_path}\n  {md_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", default="./m6_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--patch_size", type=int, default=16, choices=[8, 16, 32])
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--projection_type", default="linear", choices=["linear", "cnn", "conv3d"])
    ap.add_argument("--num_sample_windows", type=int, default=20)
    args = ap.parse_args()

    generate_m6_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        patch_size=args.patch_size, embedding_dim=args.embedding_dim,
        projection_type=args.projection_type, num_sample_windows=args.num_sample_windows,
    )
