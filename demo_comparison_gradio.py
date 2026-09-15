"""
================================================================================
Dual Model Detection Comparison Presentation Dashboard (Gradio)
================================================================================
Designed specifically for Thesis Final Review / Panel Defense.

Loads TWO trained model checkpoints simultaneously:
  1. Baseline Model (Standard Transformer -- 100% Dense Tokens, No Pruning)
  2. Proposed EDPS Model (Dynamic Sparse Patches + Transformer + Detection Head)

Does NOT require .npy annotation files to detect objects -- model predicts boxes
directly from raw event voxels! (Optional GT annotation toggle available).

Usage in Colab:
    import importlib, demo_comparison_gradio
    importlib.reload(demo_comparison_gradio)
    demo_comparison_gradio.launch(share=True)
================================================================================
"""

import os
import sys
import traceback
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches_plt
import gradio as gr

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from event_parser import EventParser
from voxel_grid import VoxelGridConfig, events_to_voxel_grid
from patch_config import PatchConfig
from patch_partition import compute_patch_grid_dims, partition_voxel_grid
from patch_statistics import compute_patch_activity_stats
from patch_adjacency import build_adjacency_map
from patch_embedding_config import PatchEmbeddingConfig
from patch_embedding_module import PatchEmbeddingModule
from edps_config import EDPSConfig
from edps_module import EDPSModule
from gated_token_wiring import compute_gated_selected_embeddings
from token_padding import pad_token_sequences
from transformer_config import TransformerEncoderConfig
from transformer_encoder import TransformerEncoder
from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead, decode_predictions


CLASS_NAMES = ["Car", "Pedestrian", "Truck", "Vehicle"]
CLASS_COLORS = ["#00D4FF", "#10B981", "#F59E0B", "#EC4899"]


def _resolve_ckpt_path(uploaded, fallback_names):
    if uploaded is not None:
        if hasattr(uploaded, "name") and os.path.exists(uploaded.name):
            return uploaded.name
        if hasattr(uploaded, "path") and os.path.exists(uploaded.path):
            return uploaded.path
        if isinstance(uploaded, str) and os.path.exists(uploaded):
            return uploaded

    # Fallback search in standard folders
    for search_dir in ["/content", "/content/checkpoints_baseline", "/content/checkpoints_edps_proposed", current_dir]:
        if os.path.exists(search_dir):
            for fname in fallback_names:
                p = os.path.join(search_dir, fname)
                if os.path.exists(p):
                    return p
                for root, _, files in os.walk(search_dir):
                    if fname in files:
                        return os.path.join(root, fname)
    return None


def _load_model_components(ckpt_path, selection_mode="normal", gate_thresh=0.50, embedding_dim=256, num_bins=10, patch_size=16, n_rows=15, n_cols=19):
    embed_cfg = PatchEmbeddingConfig(embedding_dim=embedding_dim)
    embed_mod = PatchEmbeddingModule(embed_cfg, num_bins, patch_size, n_rows, n_cols)

    edps_cfg = EDPSConfig(gate_threshold=gate_thresh, selection_mode=selection_mode)
    edps_mod = EDPSModule(edps_cfg, embedding_dim, num_bins)

    encoder_cfg = TransformerEncoderConfig(embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    encoder = TransformerEncoder(encoder_cfg)

    head_cfg = DetectionHeadConfig(embedding_dim=embedding_dim, num_classes=2)
    detection_head = SparseDetectionHead(head_cfg)

    ckpt_status = "Default Weights"
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

            if isinstance(state_dict, dict):
                if "embedding_module" in state_dict and isinstance(state_dict["embedding_module"], dict):
                    embed_mod.load_state_dict(state_dict["embedding_module"], strict=False)
                if "edps_module" in state_dict and isinstance(state_dict["edps_module"], dict):
                    edps_mod.load_state_dict(state_dict["edps_module"], strict=False)
                if "encoder" in state_dict and isinstance(state_dict["encoder"], dict):
                    encoder.load_state_dict(state_dict["encoder"], strict=False)
                if "detection_head" in state_dict and isinstance(state_dict["detection_head"], dict):
                    detection_head.load_state_dict(state_dict["detection_head"], strict=False)
                ckpt_status = f"Loaded: {os.path.basename(ckpt_path)}"
        except Exception as e:
            ckpt_status = f"Load Warning: {str(e)[:30]}"

    embed_mod.eval(); edps_mod.eval(); encoder.eval(); detection_head.eval()
    return embed_mod, edps_mod, encoder, detection_head, head_cfg, ckpt_status


def get_available_recordings():
    candidates = ["/content", "/content/gen1_local", current_dir]
    recordings = {}
    for candidate in candidates:
        if os.path.exists(candidate):
            for root, _, files in os.walk(candidate):
                for f in files:
                    if f.endswith(".dat"):
                        full_path = os.path.join(root, f)
                        display_name = f
                        if display_name not in recordings:
                            recordings[display_name] = full_path
    return recordings


def run_dual_detection_demo(selected_rec_name, baseline_ckpt_up, edps_ckpt_up, offset_ms, window_ms, gate_thresh, conf_thresh, patch_size):
    try:
        recordings = get_available_recordings()
        if not recordings:
            return None, "<div class='p-4 bg-red-900/40 text-red-300 font-bold'>Error: No .dat recordings found!</div>", "No recordings available"

        dat_path = recordings.get(selected_rec_name, list(recordings.values())[0])

        parser = EventParser()
        header = parser.reader.parse_header(dat_path)
        t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)

        window_us = int(window_ms * 1000)
        max_start_us = max(int(t_min), int(t_max - window_us))
        t_start_us = min(int(t_min + (offset_ms * 1000)), max_start_us)
        t_end_us = t_start_us + window_us

        events = parser.load_events_window(dat_path, t_start=t_start_us, t_end=t_end_us, validate=False)
        if len(events["t"]) == 0:
            return None, "<div class='p-4 bg-red-900/40 text-red-300 font-bold'>No events in window. Adjust offset slider.</div>", "Empty window"

        # Optional GT boxes for comparison if .npy file exists beside .dat
        window_boxes = []
        parent_dir = os.path.dirname(dat_path)
        stem = os.path.basename(dat_path).replace("_td.dat", "").replace(".dat", "")
        bbox_path = os.path.join(parent_dir, stem + "_bbox.npy")
        if os.path.exists(bbox_path):
            try:
                boxes = parser.load_annotations(bbox_path)
                if len(boxes) > 0 and hasattr(boxes, "dtype") and boxes.dtype.names is not None:
                    t_field = "t" if "t" in boxes.dtype.names else ("ts" if "ts" in boxes.dtype.names else boxes.dtype.names[0])
                    mask = (boxes[t_field] >= t_start_us) & (boxes[t_field] < t_end_us)
                    window_boxes = boxes[mask]
            except Exception:
                window_boxes = []

        voxel_cfg = VoxelGridConfig(num_bins=10, temporal_mode="bilinear", polarity_encoding="signed")
        grid_np = events_to_voxel_grid(
            events["t"], events["x"], events["y"], events["p"],
            t_start_us, t_end_us,
            events["header"]["height"], events["header"]["width"],
            voxel_cfg,
        )
        grid_tensor = torch.from_numpy(grid_np)

        patch_cfg = PatchConfig(patch_size=patch_size, padding_mode="zero")
        patches, meta = partition_voxel_grid(grid_tensor, patch_cfg)
        stats = compute_patch_activity_stats(patches, voxel_cfg)
        n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(
            events["header"]["height"], events["header"]["width"], patch_cfg
        )
        adj = build_adjacency_map(n_rows, n_cols, patch_cfg)

        # Resolve Checkpoints
        base_path = _resolve_ckpt_path(baseline_ckpt_up, ["best_model.pt", "best_checkpoint.pt", "checkpoints_baseline/best_model.pt"])
        edps_path = _resolve_ckpt_path(edps_ckpt_up, ["best_model.pt", "best_checkpoint.pt", "checkpoints_edps_proposed/best_model.pt"])

        # Build & Load 1. Baseline Model
        embed_b, edps_b, enc_b, head_b, head_cfg_b, status_b = _load_model_components(
            base_path, selection_mode="baseline_no_pruning", gate_thresh=0.5, n_rows=n_rows, n_cols=n_cols
        )
        # Build & Load 2. Proposed EDPS Model
        embed_e, edps_e, enc_e, head_e, head_cfg_e, status_e = _load_model_components(
            edps_path, selection_mode="normal", gate_thresh=gate_thresh, n_rows=n_rows, n_cols=n_cols
        )

        # Run Forward Pass for both models
        with torch.no_grad():
            # 1. Baseline Model Forward (100% Dense Tokens)
            emb_b, em_meta_b = embed_b(patches, meta)
            out_b = edps_b(emb_b, stats, meta, em_meta_b, adj)
            gated_b = compute_gated_selected_embeddings(out_b)
            padded_b, mask_b = pad_token_sequences([gated_b])
            encoded_b = enc_b(padded_b, attention_mask=mask_b)
            raw_b = head_b(encoded_b)
            dets_b = decode_predictions(raw_b, out_b.selected_metadata, head_cfg_b, attention_mask=mask_b, batch_index=0)

            # 2. Proposed EDPS Model Forward (Dynamic Sparse Tokens)
            emb_e, em_meta_e = embed_e(patches, meta)
            out_e = edps_e(emb_e, stats, meta, em_meta_e, adj)
            gated_e = compute_gated_selected_embeddings(out_e)
            padded_e, mask_e = pad_token_sequences([gated_e])
            encoded_e = enc_e(padded_e, attention_mask=mask_e)
            raw_e = head_e(encoded_e)
            dets_e = decode_predictions(raw_e, out_e.selected_metadata, head_cfg_e, attention_mask=mask_e, batch_index=0)

        n_total = len(out_b.importance_scores)
        n_kept_e = int(out_e.binary_mask.sum().item())
        keep_pct = (n_kept_e / n_total) * 100.0
        pruned_pct = 100.0 - keep_pct
        flops_red = (n_total * n_total) / max(1.0, n_kept_e * n_kept_e)

        # Render High-Resolution Side-by-Side Plots
        bg_col = "#0A0F1E"
        card_col = "#0F172A"
        border_col = "#1E293B"
        cyan = "#00D4FF"
        green = "#10B981"
        red = "#EF4444"
        text_col = "#F1F5F9"
        muted_col = "#94A3B8"

        plt.rcParams.update({
            "figure.facecolor": bg_col, "axes.facecolor": card_col,
            "text.color": text_col, "axes.labelcolor": muted_col,
            "xtick.color": muted_col, "ytick.color": muted_col,
            "axes.edgecolor": border_col, "font.family": "DejaVu Sans",
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150, facecolor=bg_col)
        fig.patch.set_facecolor(bg_col)

        event_raw = np.sum(np.abs(grid_np), axis=0)
        event_img = np.log1p(event_raw)
        vmax = np.percentile(event_img[event_img > 0], 99) if np.any(event_img > 0) else 1.0

        # --- LEFT PANEL: Baseline Dense Model ---
        ax1.set_facecolor(card_col)
        ax1.imshow(event_img, cmap="gray_r", origin="upper", vmin=0, vmax=vmax, interpolation="nearest")
        ax1.set_title("Standard Transformer Baseline (Dense -- 100% Tokens)", fontsize=11, fontweight="bold", color=red, pad=10)
        ax1.text(0.99, 0.97, f"{n_total} / {n_total} tokens (100% FLOPs)", transform=ax1.transAxes, ha="right", va="top", fontsize=8, color=red, fontweight="bold")

        # Baseline grid patches
        for m in meta:
            ax1.add_patch(patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.4, edgecolor="#3B82F6", facecolor="none", alpha=0.3))

        # Baseline Predicted Bounding Boxes
        shown_dets_b = [d for d in dets_b if d.combined_score >= conf_thresh]
        for d in shown_dets_b:
            color = CLASS_COLORS[d.predicted_class % len(CLASS_COLORS)]
            w_box, h_box = max(d.box_x1 - d.box_x0, 10), max(d.box_y1 - d.box_y0, 10)
            ax1.add_patch(patches_plt.Rectangle((d.box_x0, d.box_y0), w_box, h_box, linewidth=2.0, edgecolor=color, facecolor="none"))
            cls_name = CLASS_NAMES[d.predicted_class % len(CLASS_NAMES)]
            ax1.text(d.box_x0, max(d.box_y0 - 3, 0), f"{cls_name}:{d.combined_score:.2f}", color=color, fontsize=7, fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.1", facecolor="#0F172A", edgecolor=color, linewidth=0.5))

        ax1.set_axis_off()
        ax1.annotate(f"Baseline Predictions: {len(shown_dets_b)} objects detected (Dense 285 tokens)", xy=(0.5, -0.02), xycoords="axes fraction", ha="center", fontsize=8, color=muted_col)

        # --- RIGHT PANEL: Proposed EDPS Sparse Model ---
        ax2.set_facecolor(card_col)
        ax2.imshow(event_img, cmap="gray_r", origin="upper", vmin=0, vmax=vmax, interpolation="nearest")
        ax2.set_title(f"Proposed EDPS Model (Dynamic Sparse -- {n_kept_e}/{n_total} tokens, {pruned_pct:.1f}% pruned)", fontsize=11, fontweight="bold", color=green, pad=10)
        ax2.text(0.99, 0.97, f"{flops_red:.1f}x Attention FLOPs Reduction", transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=cyan, fontweight="bold")

        # EDPS Retained Green Patches vs Red Pruned
        mask_e_np = out_e.binary_mask.numpy()
        scores_e_np = out_e.importance_scores.numpy()
        for i, m in enumerate(meta):
            if mask_e_np[i]:
                ax2.add_patch(patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=1.2, edgecolor=green, facecolor=green, alpha=0.20))
            else:
                ax2.add_patch(patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.4, edgecolor=red, facecolor=red, alpha=0.06))

        # EDPS Predicted Bounding Boxes
        shown_dets_e = [d for d in dets_e if d.combined_score >= conf_thresh]
        for d in shown_dets_e:
            color = CLASS_COLORS[d.predicted_class % len(CLASS_COLORS)]
            w_box, h_box = max(d.box_x1 - d.box_x0, 10), max(d.box_y1 - d.box_y0, 10)
            ax2.add_patch(patches_plt.Rectangle((d.box_x0, d.box_y0), w_box, h_box, linewidth=2.2, edgecolor=color, facecolor="none"))
            cls_name = CLASS_NAMES[d.predicted_class % len(CLASS_NAMES)]
            ax2.text(d.box_x0, max(d.box_y0 - 3, 0), f"{cls_name}:{d.combined_score:.2f} (EDPS)", color=color, fontsize=7, fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.1", facecolor="#0F172A", edgecolor=color, linewidth=0.5))

        # Ground Truth Dashed Gold Boxes (Reference)
        for b in window_boxes:
            bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            ax2.add_patch(patches_plt.Rectangle((bx, by), bw, bh, linewidth=1.2, edgecolor="#F59E0B", linestyle="--", facecolor="none", alpha=0.5))

        ax2.set_axis_off()
        ax2.annotate(f"EDPS Predictions: {len(shown_dets_e)} objects detected | {pruned_pct:.1f}% Tokens Pruned | Dashed=GT", xy=(0.5, -0.02), xycoords="axes fraction", ha="center", fontsize=8, color=muted_col)

        plt.tight_layout(pad=2.0)

        # Summary HTML Table
        html_kpi = f"""
        <div style="display:flex;gap:12px;margin-top:10px;justify-content:space-between">
          <div style="background:#0F172A;border:1px solid #1E293B;padding:12px;border-radius:10px;flex:1 text-align:center">
            <div style="color:#94A3B8;font-size:11px">Baseline Tokens</div>
            <div style="color:#EF4444;font-size:20px;font-weight:bold">{n_total} / {n_total} (100%)</div>
          </div>
          <div style="background:#0F172A;border:1px solid #1E293B;padding:12px;border-radius:10px;flex:1 text-align:center">
            <div style="color:#94A3B8;font-size:11px">EDPS Retained Tokens</div>
            <div style="color:#10B981;font-size:20px;font-weight:bold">{n_kept_e} / {n_total} ({keep_pct:.1f}%)</div>
          </div>
          <div style="background:#0F172A;border:1px solid #1E293B;padding:12px;border-radius:10px;flex:1 text-align:center">
            <div style="color:#94A3B8;font-size:11px">FLOPs Acceleration</div>
            <div style="color:#00D4FF;font-size:20px;font-weight:bold">{flops_red:.2f}x Speedup</div>
          </div>
          <div style="background:#0F172A;border:1px solid #1E293B;padding:12px;border-radius:10px;flex:1 text-align:center">
            <div style="color:#94A3B8;font-size:11px">Object Detections</div>
            <div style="color:#F59E0B;font-size:20px;font-weight:bold">{len(shown_dets_e)} EDPS vs {len(shown_dets_b)} Base</div>
          </div>
        </div>
        """

        status_str = f"Loaded Baseline ({status_b}) | Loaded EDPS ({status_e}) | Window [{t_start_us/1e6:.2f}s, {t_end_us/1e6:.2f}s]"
        return fig, html_kpi, status_str

    except Exception as e:
        err = f"Execution Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, f"<div style='color:red'>{err}</div>", "Error"


def launch(share: bool = True):
    recordings_map = get_available_recordings()
    rec_choices = list(recordings_map.keys())

    with gr.Blocks(title="EDPS Thesis Defense Presentation -- Dual Model Detection Showcase", theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate")) as demo:
        gr.HTML("""
        <div style="background:#0F172A;border:1px solid #1E293B;padding:16px;border-radius:12px;margin-bottom:16px">
          <h1 style="color:#F1F5F9;font-size:20px;font-weight:bold;margin:0">Dual Model Object Detection Showcase (Baseline vs EDPS)</h1>
          <p style="color:#94A3B8;font-size:12px;margin-top:4px">
            Final Project Thesis Review Demonstration &bull; Evaluates real 25-epoch model predicted bounding boxes and dynamic patch pruning side-by-side.
          </p>
        </div>
        """)

        with gr.Row():
            rec_dropdown = gr.Dropdown(choices=rec_choices, value=rec_choices[0] if rec_choices else None, label="Recording Stream (.dat)")
            baseline_ckpt = gr.File(label="1. Baseline Checkpoint (.pt)", type="filepath")
            edps_ckpt = gr.File(label="2. EDPS Checkpoint (.pt)", type="filepath")

        run_btn = gr.Button("⚡ RUN DUAL MODEL DETECTION SHOWCASE", variant="primary")

        with gr.Row():
            with gr.Column(scale=3):
                offset_slider = gr.Slider(minimum=0, maximum=55000, value=500, step=500, label="Time Offset (ms)")
                window_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Window Duration (ms)")
                gate_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.50, step=0.05, label="EDPS Gate Threshold (tau)")
                conf_slider = gr.Slider(minimum=0.05, maximum=0.80, value=0.15, step=0.05, label="Detection Score Cutoff")
                patch_size_dropdown = gr.Dropdown(choices=[16, 32], value=16, label="Patch Size (px)")

            with gr.Column(scale=9):
                plot_out = gr.Plot(label="")

        kpi_out = gr.HTML("")
        status_out = gr.Textbox(label="System Status", interactive=False)

        run_btn.click(
            fn=run_dual_detection_demo,
            inputs=[rec_dropdown, baseline_ckpt, edps_ckpt, offset_slider, window_slider, gate_slider, conf_slider, patch_size_dropdown],
            outputs=[plot_out, kpi_out, status_out],
        )

    demo.launch(share=share)


if __name__ == "__main__":
    launch(share=True)
