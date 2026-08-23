"""
================================================================================
M7 Research Log Report Generator
================================================================================
Runs the M7 unit test suite, computes retained-patch/token-reduction/FLOPs
statistics over sampled windows, the collapse diagnostic, gate-score
logging (refinement 4), saves an example EDPS visualization, and writes
M7_report.json / M7_report.md.

Usage:
    python generate_m7_report.py --dataset_root /path/to/dataset --output_dir /content/m7_report
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
from edps_config import EDPSConfig
from edps_module import EDPSModule
from edps_diagnostics import run_collapse_diagnostic
from edps_cost_analysis import compute_edps_flops_comparison
from edps_visualization import visualize_edps
from gate_score_logger import GateScoreLogger
from test_m7_edps import run_all_m7_tests


def generate_m7_report(
    dataset_root: str,
    output_dir: str = ".",
    window_us: int = 50_000,
    embedding_dim: int = 256,
    num_sample_windows: int = 20,
    seed: int = 42,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    all_passed, test_results = run_all_m7_tests(dataset_root, window_us=window_us)

    voxel_config = VoxelGridConfig()
    patch_config = PatchConfig(patch_size=16)
    embedding_config = PatchEmbeddingConfig(embedding_dim=embedding_dim)
    edps_config = EDPSConfig()

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
    edps_module = EDPSModule(edps_config, embedding_dim=embedding_dim, num_bins=voxel_config.num_bins)

    n_sample = min(num_sample_windows, len(dataset))
    gate_logger = GateScoreLogger()
    retained_counts, removed_counts = [], []
    clamp_triggers = []
    all_scores_for_diag = []
    all_densities_for_diag = []
    example_out = None
    example_patch_result = None

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
            embeddings, embedding_meta = embedding_module(patch_result.patches, patch_result.metadata)
            out = edps_module(embeddings, patch_result.stats, patch_result.metadata, embedding_meta, patch_result.adjacency)

        gate_logger.log(out.importance_scores, threshold=edps_config.gate_threshold, step=i)
        retained_counts.append(out.token_reduction_stats["n_retained"])
        removed_counts.append(out.token_reduction_stats["n_removed"])
        clamp_triggers.append(out.token_reduction_stats["clamp_triggered"])
        all_scores_for_diag.append(out.importance_scores.numpy())
        all_densities_for_diag.append(np.array([s.event_density for s in patch_result.stats]))

        if example_out is None:
            example_out = out
            example_patch_result = patch_result

    n_input = example_patch_result.patches.shape[0]
    avg_retained = float(np.mean(retained_counts))
    avg_removed = float(np.mean(removed_counts))
    avg_token_reduction_pct = 100.0 * avg_removed / n_input if n_input else 0.0

    diag = run_collapse_diagnostic(
        np.concatenate(all_scores_for_diag), np.concatenate(all_densities_for_diag),
        warn_threshold=edps_config.collapse_correlation_warn_threshold,
    )

    flops_comparison = compute_edps_flops_comparison(
        n_input_tokens=n_input, n_retained_tokens=int(round(avg_retained)), embedding_dim=embedding_dim,
    )

    viz_path = os.path.join(output_dir, "edps_visualization_example.png")
    visualize_edps(
        example_out.importance_scores, example_out.binary_mask, example_patch_result.metadata,
        example_patch_result.n_rows, example_patch_result.n_cols, edps_config.gate_threshold,
        example_out.token_reduction_stats, out_path=viz_path,
    )

    gate_hist_path = os.path.join(output_dir, "gate_score_evolution.png")
    gate_logger.plot_histogram(gate_hist_path)
    gate_log_json_path = os.path.join(output_dir, "gate_score_log.json")
    gate_logger.save(gate_log_json_path)

    n_params = count_parameters(edps_module)
    elapsed_total = time.time() - start_time

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "module": "M7 - Event-Aware Dynamic Patch Selection (EDPS)",
        "config": {
            "gate_threshold": edps_config.gate_threshold,
            "clamp_min_ratio": edps_config.clamp_min_ratio,
            "clamp_max_ratio": edps_config.clamp_max_ratio,
            "sparsity_min_ratio": edps_config.sparsity_min_ratio,
            "sparsity_max_ratio": edps_config.sparsity_max_ratio,
            "lambda_sparsity": edps_config.lambda_sparsity,
            "use_neighbor_context": edps_config.use_neighbor_context,
            "local_feature_dim": edps_config.local_feature_dim,
            "scoring_hidden_dim": edps_config.scoring_hidden_dim,
        },
        "retained_patch_statistics": {
            "num_windows_sampled": n_sample,
            "n_input_tokens_per_window": n_input,
            "avg_retained_tokens": avg_retained,
            "std_retained_tokens": float(np.std(retained_counts)),
            "avg_removed_tokens": avg_removed,
            "avg_token_reduction_pct": avg_token_reduction_pct,
            "clamp_triggered_fraction_of_windows": float(np.mean(clamp_triggers)),
        },
        "collapse_diagnostic": diag,
        "flops_comparison": flops_comparison,
        "gate_score_summary": gate_logger.summary(),
        "parameter_count": n_params,
        "example_visualization": viz_path,
        "gate_score_evolution_plot": gate_hist_path,
        "gate_score_log": gate_log_json_path,
        "unit_test_results": [
            {"name": name, "passed": bool(passed), "detail": detail}
            for name, passed, detail in test_results
        ],
        "all_unit_tests_passed": bool(all_passed),
        "execution_time_seconds": round(elapsed_total, 4),
        "important_caveat": (
            "The EDPS scoring network is UNTRAINED at this stage (no detection loss "
            "exists until M8-M10 are built). All retained-token, FLOPs-reduction, and "
            "collapse-diagnostic numbers in this report reflect a randomly-initialized "
            "scorer's behavior, validated for MECHANISM correctness (gating, clamping, "
            "neighbor-awareness, gradient flow), not for learned importance quality. "
            "Re-run this report after training for numbers that reflect real behavior."
        ),
    }

    json_path = os.path.join(output_dir, "M7_report.json")
    md_path = os.path.join(output_dir, "M7_report.md")

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write("# M7 - Event-Aware Dynamic Patch Selection (EDPS) Report\n\n")
        f.write(f"Generated: {report['generated_at_utc']}\n\n")
        f.write(f"> **{report['important_caveat']}**\n\n")

        f.write("## Configuration\n\n")
        for k, v in report["config"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Retained Patch Statistics (aggregated over {n_sample} sampled windows)\n\n")
        for k, v in report["retained_patch_statistics"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Collapse Diagnostic (score vs. event density)\n\n")
        for k, v in diag.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## FLOPs Comparison (Transformer, theoretical projection)\n\n")
        for k, v in flops_comparison.items():
            if k != "assumptions":
                f.write(f"- **{k}**: {v}\n")
        f.write(f"- **assumptions**: {flops_comparison['assumptions']}\n")

        f.write("\n## Gate Score Summary\n\n")
        for k, v in report["gate_score_summary"].items():
            f.write(f"- **{k}**: {v}\n")

        f.write(f"\n## Parameter Count\n\n- **total_parameters**: {n_params}\n")

        f.write(f"\n## Visualizations\n\n- `{viz_path}`\n- `{gate_hist_path}`\n- gate score log (json): `{gate_log_json_path}`\n")

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
    ap.add_argument("--output_dir", default="./m7_report")
    ap.add_argument("--window_us", type=int, default=50_000)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--num_sample_windows", type=int, default=20)
    args = ap.parse_args()

    generate_m7_report(
        args.dataset_root, args.output_dir, window_us=args.window_us,
        embedding_dim=args.embedding_dim, num_sample_windows=args.num_sample_windows,
    )
