"""
================================================================================
Unit Tests for M8 (Transformer Encoder)
================================================================================
Covers: output shape correctness, PADDING MASK ISOLATION (the safety-
critical test -- padded tokens must not leak into real token outputs),
attention weight return (shape + valid probability distribution), Pre-LN
gradient flow, determinism, no detection head (output dim == embedding_dim,
not num_classes), dataset integration, DataLoader integration, attention
visualization, report generation.

Run standalone:
    python test_m8_transformer.py --dataset_root path/to/dataset
================================================================================
"""

import os
import argparse
import torch

from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder, count_parameters
from token_padding import pad_token_sequences
from attention_visualization import visualize_attention_maps
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims
from dataset_split import create_split
from gen1_dataset import EDPSGen1Dataset
from m8_transformer_collate import build_full_pipeline_transformer_dataloader


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


def run_all_m8_tests(dataset_root: str, window_us: int = 200_000, cleanup: bool = True):
    tr = TestResult()
    print(f"Running M8 transformer encoder unit tests against: {dataset_root}\n")

    cfg = TransformerEncoderConfig()

    # ---- 1. Output shape correctness ----
    try:
        enc = TransformerEncoder(cfg)
        tokens = torch.randn(2, 10, cfg.embedding_dim)
        out = enc(tokens)
        ok = out.shape == tokens.shape
        tr.record("output_shape_correctness", ok, f"shape={tuple(out.shape)}")
    except Exception as e:
        tr.record("output_shape_correctness", False, str(e))

    # ---- 2. Padding mask isolation (THE critical test) ----
    try:
        torch.manual_seed(0)
        enc2 = TransformerEncoder(cfg)
        enc2.eval()
        real_tokens = torch.randn(5, cfg.embedding_dim)
        with torch.no_grad():
            out_no_pad = enc2(real_tokens.unsqueeze(0))
            padded_list = [real_tokens, torch.randn(2, cfg.embedding_dim)]
            padded, mask = pad_token_sequences(padded_list)
            out_padded = enc2(padded, attention_mask=mask)
        max_diff = (out_padded[0, :5, :] - out_no_pad[0]).abs().max().item()
        ok = max_diff < 1e-4
        tr.record("padding_mask_isolation", ok, f"max_diff={max_diff:.2e} (padded tokens must not leak)")
    except Exception as e:
        tr.record("padding_mask_isolation", False, str(e))

    # ---- 3. Attention weight return: shape + valid probability distribution ----
    try:
        enc3 = TransformerEncoder(cfg)
        enc3.eval()
        tokens3 = torch.randn(2, 10, cfg.embedding_dim)
        with torch.no_grad():
            out3, attn_maps = enc3(tokens3, return_attention=True)
        shape_ok = (len(attn_maps) == cfg.num_layers
                    and attn_maps[0].shape == (2, cfg.num_heads, 10, 10))
        row_sums = attn_maps[0].sum(dim=-1)
        prob_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
        tr.record("attention_weights_correctness", shape_ok and prob_ok,
                   f"shape_ok={shape_ok}, rows_sum_to_1={prob_ok}")
    except Exception as e:
        tr.record("attention_weights_correctness", False, str(e))

    # ---- 4. Gradient flow (Pre-LN residual blocks) ----
    try:
        enc4 = TransformerEncoder(cfg)
        tokens4 = torch.randn(2, 10, cfg.embedding_dim, requires_grad=True)
        out4 = enc4(tokens4)
        out4.sum().backward()
        input_grad_ok = tokens4.grad is not None and tokens4.grad.abs().sum() > 0
        param_grad_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc4.parameters())
        tr.record("gradient_flow", input_grad_ok and param_grad_ok,
                   f"input_grad={input_grad_ok}, param_grad={param_grad_ok}")
    except Exception as e:
        tr.record("gradient_flow", False, str(e))

    # ---- 5. Determinism in eval mode ----
    try:
        enc5 = TransformerEncoder(cfg)
        enc5.eval()
        tokens5 = torch.randn(2, 10, cfg.embedding_dim)
        with torch.no_grad():
            out_a = enc5(tokens5)
            out_b = enc5(tokens5)
        ok = torch.allclose(out_a, out_b)
        tr.record("determinism_eval_mode", ok)
    except Exception as e:
        tr.record("determinism_eval_mode", False, str(e))

    # ---- 6. No detection head: output dim == embedding_dim ----
    try:
        enc6 = TransformerEncoder(cfg)
        tokens6 = torch.randn(2, 7, cfg.embedding_dim)
        out6 = enc6(tokens6)
        ok = out6.shape[-1] == cfg.embedding_dim
        tr.record("no_detection_head_present", ok, f"output_dim={out6.shape[-1]} == embedding_dim={cfg.embedding_dim}")
    except Exception as e:
        tr.record("no_detection_head_present", False, str(e))

    # ---- 7. Dataset integration ----
    dataset = None
    try:
        manifest = create_split(dataset_root, seed=42)
        dataset = EDPSGen1Dataset(dataset_root, manifest, split="train", window_us=window_us)
        n_rows, n_cols, _, _ = compute_patch_grid_dims(
            dataset[0]["sensor_height"], dataset[0]["sensor_width"], PatchConfig(patch_size=16)
        )
        tr.record("dataset_integration", True, f"grid=({n_rows},{n_cols})")
    except Exception as e:
        tr.record("dataset_integration", False, str(e))

    # ---- 8. DataLoader integration ----
    batch = None
    if dataset is not None:
        try:
            loader = build_full_pipeline_transformer_dataloader(
                dataset, dataset_root, n_rows, n_cols, batch_size=3, shuffle=False, return_attention=True,
            )
            batch = next(iter(loader))
            ok = (batch["encoded_tokens"].ndim == 3
                  and batch["encoded_tokens"].shape[0] == min(3, len(dataset))
                  and batch["encoded_tokens"].shape[-1] == cfg.embedding_dim
                  and len(batch["attention_maps"]) == cfg.num_layers)
            tr.record("dataloader_integration", ok, f"encoded_tokens.shape={tuple(batch['encoded_tokens'].shape)}")
        except Exception as e:
            tr.record("dataloader_integration", False, str(e))
    else:
        tr.record("dataloader_integration", False, "skipped: no dataset")

    # ---- 9. Attention visualization ----
    if batch is not None:
        try:
            n_valid = int(batch["attention_mask"][0].sum().item())
            out_path = "test_m8_attn_viz.png"
            saved = visualize_attention_maps(batch["attention_maps"], sample_index=0,
                                              n_valid_tokens=n_valid, out_path=out_path)
            ok = os.path.exists(saved) and os.path.getsize(saved) > 0
            tr.record("visualization", ok, f"saved to {saved}")
            if cleanup and ok:
                os.remove(saved)
        except Exception as e:
            tr.record("visualization", False, str(e))
    else:
        tr.record("visualization", False, "skipped: no batch")

    # ---- 10. Report generation (importability smoke test) ----
    try:
        from generate_m8_report import generate_m8_report  # noqa
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

    passed, _ = run_all_m8_tests(args.dataset_root, window_us=args.window_us)
    raise SystemExit(0 if passed else 1)
