"""
================================================================================
M9 Research Log Report Generator
================================================================================
Runs the M9 unit test suite, measures detection head cost statistics
(params/FLOPs/latency/memory), saves an example detection visualization,
and writes M9_report.json / M9_report.md.
================================================================================
"""

import os
import json
import time
import argparse
from datetime import datetime, timezone

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
from patch_embedding_module import PatchEmbeddingModule
from edps_config import EDPSConfig
from edps_module import EDPSModule
from token_padding import pad_token_sequences
from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder
from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead, decode_predictions, count_parameters
from detection_head_cost_analysis import measure_detection_head_cost
from detection_visualization import visualize_detections
from test_m9_detection_head import run_all_m9_tests


def generate_m9_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    num_sample_windows: int = 10,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_m9_tests(dataset_root, window_us=window_us)

    voxel_config = VoxelGridConfig()
    patch_config = PatchConfig(patch_size=16)
    embedding_config = PatchEmbeddingConfig()
    edps_config = EDPSConfig()
    transformer_config = TransformerEncoderConfig()
    detection_config = DetectionHeadConfig(embedding_dim=embedding_config.embedding_dim)

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
    edps_module = EDPSModule(edps_config, embedding_dim=embedding_config.embedding_dim, num_bins=voxel_config.num_bins)
    encoder = TransformerEncoder(transformer_config)
    detection_head = SparseDetectionHead(detection_config)

    n_sample = min(num_sample_windows, len(dataset))
    selected_embeddings_list = []
    selected_metadata_list = []
    example_patch_result = None
    example_sample = None

    for i in range(n_sample):
        sample = dataset[i]
        cleaned = preprocessor.process(sample, dataset_root=dataset_root)
        grid = voxel_gen.generate({
            "t": cleaned["t"], "x": cleaned["x"], "y": cleaned["y"], "p": cleaned["p"],
            "t_start": cleaned["t_start"], "t_end": cleaned["t_end"],
            "sensor_height": cleaned["sensor_height"], "sensor_width": cleaned["sensor_width"],
        })
        patch_result = patch_gen.generate(grid)
        with torch.no_grad():
            embeddings, embedding_meta = embedding_module(patch_result.patches, patch_result.metadata)
            edps_out = edps_module(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)
        selected_embeddings_list.append(edps_out.selected_embeddings)
        selected_metadata_list.append(edps_out.selected_metadata)
        if example_patch_result is None:
            example_patch_result = patch_result
            example_sample = cleaned

    padded, attention_mask = pad_token_sequences(selected_embeddings_list)
    with torch.no_grad():
        encoded = encoder(padded, attention_mask=attention_mask)
        raw_predictions = detection_head(encoded)

    example_dets = decode_predictions(
        raw_predictions, selected_metadata_list[0], detection_config, attention_mask=attention_mask, batch_index=0
    )

    cost = measure_detection_head_cost(detection_head, encoded)

    viz_path = os.path.join(output_dir, "detection_visualization_example.png")
    visualize_detections(
        example_dets, example_patch_result.metadata, selected_metadata_list[0], n_rows, n_cols,
        example_sample["sensor_height"], example_sample["sensor_width"], out_path=viz_path,
    )

    n_params = count_parameters(detection_head)
    avg_tokens = float(sum(e.shape[0] for e in selected_embeddings_list) / len(selected_embeddings_list))

    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M9 - Sparse Anchor-Free Per-Token Detection Head",
        "config": {
            "embedding_dim": detection_config.embedding_dim,
            "hidden_dim": detection_config.hidden_dim,
            "num_classes": detection_config.num_classes,
            "dropout": detection_config.dropout,
            "distance_scale": detection_config.distance_scale,
        },
        "parameter_count": n_params,
        "computational_cost": cost,
        "avg_retained_tokens_sampled": avg_tokens,
        "example_num_detections": len(example_dets),
        "example_visualization": viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
        "note": (
            "No training/loss exists yet (M10). Predictions here come from a "
            "randomly-initialized head -- validated for ARCHITECTURE and "
            "PLUMBING correctness (shapes, decode math, metadata mapping, "
            "gradient flow), not for detection quality."
        ),
    }

    json_path = os.path.join(output_dir, "M9_report.json")
    md_path = os.path.join(output_dir, "M9_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M9 - Sparse Anchor-Free Per-Token Detection Head Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")
        f.write(f"> {report['note']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Parameter Count\n\n- **total_parameters**: {n_params}\n")

        f.write("\n## Computational Cost (measured)\n\n")
        for k, v in cost.items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Example\n\n- Average retained tokens (sampled): {avg_tokens:.1f}\n")
        f.write(f"- Detections decoded for example sample: {len(example_dets)}\n")
        f.write(f"- Visualization: `{viz_path}`\n")

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
    ap.add_argument("--output_dir", default="./m9_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--num_sample_windows", type=int, default=10)
    args = ap.parse_args()

    generate_m9_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        num_sample_windows=args.num_sample_windows,
    )
