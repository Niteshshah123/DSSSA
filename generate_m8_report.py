"""
================================================================================
M8 Research Log Report Generator
================================================================================
Runs the M8 unit test suite, measures real transformer cost statistics
(params/FLOPs/latency/memory), computes the before/after EDPS FLOPs and
LATENCY comparison (latency now measured for real, unlike M7's pure
projection), saves an example attention map visualization, and writes
M8_report.json / M8_report.md.
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
from transformer_encoder import TransformerEncoder, count_parameters
from transformer_cost_analysis import measure_transformer_cost, compare_before_after_edps
from attention_visualization import visualize_attention_maps
from test_m8_transformer import run_all_m8_tests


def generate_m8_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    num_sample_windows: int = 10,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_m8_tests(dataset_root, window_us=window_us)

    voxel_config = VoxelGridConfig()
    patch_config = PatchConfig(patch_size=16)
    embedding_config = PatchEmbeddingConfig()
    edps_config = EDPSConfig()
    transformer_config = TransformerEncoderConfig()

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

    n_sample = min(num_sample_windows, len(dataset))
    selected_embeddings_list = []
    n_input_tokens = n_rows * n_cols

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

    padded, attention_mask = pad_token_sequences(selected_embeddings_list)
    with torch.no_grad():
        encoded, attn_maps = encoder(padded, attention_mask=attention_mask, return_attention=True)

    avg_retained = float(sum(e.shape[0] for e in selected_embeddings_list) / len(selected_embeddings_list))

    cost = measure_transformer_cost(encoder, padded, attention_mask=attention_mask)
    comparison = compare_before_after_edps(
        encoder, n_input_tokens=n_input_tokens, n_retained_tokens=int(round(avg_retained)),
        embedding_dim=embedding_config.embedding_dim,
    )

    viz_path = os.path.join(output_dir, "attention_maps_example.png")
    n_valid = int(attention_mask[0].sum().item())
    visualize_attention_maps(attn_maps, sample_index=0, n_valid_tokens=n_valid, out_path=viz_path)

    n_params = count_parameters(encoder)
    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M8 - Transformer Encoder",
        "config": {
            "embedding_dim": transformer_config.embedding_dim,
            "num_heads": transformer_config.num_heads,
            "num_layers": transformer_config.num_layers,
            "ffn_dim": transformer_config.ffn_dim,
            "dropout": transformer_config.dropout,
            "norm_style": transformer_config.norm_style,
        },
        "parameter_count": n_params,
        "computational_cost": cost,
        "before_after_edps_comparison": comparison,
        "avg_retained_tokens_sampled": avg_retained,
        "n_input_tokens_full_grid": n_input_tokens,
        "example_visualization": viz_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
        "note": (
            "No detection head exists yet (M9). This report characterizes the "
            "Transformer ENCODER only -- FLOPs/latency reflect encoding cost, "
            "not full detection pipeline cost."
        ),
    }

    json_path = os.path.join(output_dir, "M8_report.json")
    md_path = os.path.join(output_dir, "M8_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M8 - Transformer Encoder Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")
        f.write(f"> {report['note']}\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Parameter Count\n\n- **total_parameters**: {n_params}\n")

        f.write("\n## Computational Cost (measured)\n\n")
        for k, v in cost.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Before/After EDPS Comparison\n\n")
        for k, v in comparison.items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Example Visualization\n\n- `{viz_path}`\n")

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
    ap.add_argument("--output_dir", default="./m8_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--num_sample_windows", type=int, default=10)
    args = ap.parse_args()

    generate_m8_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        num_sample_windows=args.num_sample_windows,
    )
