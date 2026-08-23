"""
================================================================================
M10 EDPS Interactive Panel Demo Dashboard (Gradio Version for Colab)
================================================================================
Features:
  1. Print Patch Score Values directly inside each patch box (0.00 to 1.00).
  2. Automatic Scan of /content/ for .dat files (0s Load Time!).
  3. Defensive NumPy Structured Annotation Filtering.
  4. Automatic Timestamp Synchronization.
  5. Live EDPS Gate Threshold Slider (0.10 to 0.90).
  6. Side-by-Side Comparison (Baseline 100% Tokens vs Proposed EDPS Dynamic Tokens).
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

# Ensure project modules are importable across Colab subdirectories
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [current_dir, os.getcwd(), "/content"]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

for search_root in ["/content", "/content/drive", current_dir, os.getcwd()]:
    if os.path.exists(search_root):
        for root, _, files in os.walk(search_root):
            if "event_parser.py" in files:
                if root not in sys.path:
                    sys.path.insert(0, root)
                break

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


def process_demo(selected_rec_name, uploaded_dat, uploaded_bbox, uploaded_ckpt, offset_ms, window_ms, gate_thresh, patch_size, show_scores):
    try:
        # Determine DAT file path
        if uploaded_dat is not None:
            if isinstance(uploaded_dat, dict):
                dat_path = uploaded_dat.get("name") or uploaded_dat.get("path")
            elif hasattr(uploaded_dat, "name"):
                dat_path = uploaded_dat.name
            else:
                dat_path = str(uploaded_dat)
        elif selected_rec_name:
            rec_map = get_available_recordings()
            dat_path = rec_map.get(selected_rec_name)
        else:
            return None, "⚠️ Please select a recording from the dropdown or upload a `.dat` file."

        if not dat_path or not os.path.exists(dat_path):
            return None, f"⚠️ DAT file not found at: {dat_path}"

        # Determine BBOX file path
        bbox_path = None
        if uploaded_bbox is not None:
            if isinstance(uploaded_bbox, dict):
                bbox_path = uploaded_bbox.get("name") or uploaded_bbox.get("path")
            elif hasattr(uploaded_bbox, "name"):
                bbox_path = uploaded_bbox.name
            else:
                bbox_path = str(uploaded_bbox)
        else:
            possible_bbox = dat_path.replace("_td.dat", "_bbox.npy").replace(".dat", ".npy")
            if os.path.exists(possible_bbox):
                bbox_path = possible_bbox
            else:
                parent_dir = os.path.dirname(dat_path)
                stem = os.path.basename(dat_path).replace("_td.dat", "").replace(".dat", "")
                alt_bbox = os.path.join(parent_dir, stem + "_bbox.npy")
                if os.path.exists(alt_bbox):
                    bbox_path = alt_bbox

        parser = EventParser()
        header = parser.reader.parse_header(dat_path)
        t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)

        # Start window from t_min + offset_ms
        t_start_us = int(t_min + (offset_ms * 1000))
        t_end_us = t_start_us + int(window_ms * 1000)
        if t_end_us > t_max:
            t_start_us = int(t_min)
            t_end_us = t_start_us + int(window_ms * 1000)

        events = parser.load_events_window(dat_path, t_start=t_start_us, t_end=t_end_us, validate=False)
        if len(events["t"]) == 0:
            return None, f"⚠️ No events found in timestamp window [{t_start_us} us, {t_end_us} us]. Try setting Offset to 0."

        # Defensive structured array filtering for annotations
        window_boxes = []
        if bbox_path and os.path.exists(bbox_path):
            try:
                boxes = parser.load_annotations(bbox_path)
                if len(boxes) > 0 and hasattr(boxes, "dtype") and boxes.dtype.names is not None:
                    t_field = "t" if "t" in boxes.dtype.names else ("ts" if "ts" in boxes.dtype.names else boxes.dtype.names[0])
                    mask = (boxes[t_field] >= t_start_us) & (boxes[t_field] < t_end_us)
                    window_boxes = boxes[mask]
            except Exception as e:
                window_boxes = []

        voxel_cfg = VoxelGridConfig(num_bins=10, temporal_mode="bilinear", polarity_encoding="signed")
        grid_np = events_to_voxel_grid(
            events["t"], events["x"], events["y"], events["p"],
            t_start_us, t_end_us, events["header"]["height"], events["header"]["width"], voxel_cfg
        )
        grid_tensor = torch.from_numpy(grid_np)

        patch_cfg = PatchConfig(patch_size=patch_size, padding_mode="zero")
        patches, meta = partition_voxel_grid(grid_tensor, patch_cfg)
        stats = compute_patch_activity_stats(patches, voxel_cfg)
        n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(events["header"]["height"], events["header"]["width"], patch_cfg)
        adj = build_adjacency_map(n_rows, n_cols, patch_cfg)

        embed_cfg = PatchEmbeddingConfig(embedding_dim=128)
        embed_mod = PatchEmbeddingModule(embed_cfg, voxel_cfg.num_channels, patch_size, n_rows, n_cols)
        edps_cfg = EDPSConfig(gate_threshold=gate_thresh, selection_mode="normal")
        edps_mod = EDPSModule(edps_cfg, embed_cfg.embedding_dim, voxel_cfg.num_bins)

        # Auto-detect or load uploaded trained checkpoint (.pt)
        ckpt_path = None
        if uploaded_ckpt is not None:
            if isinstance(uploaded_ckpt, dict):
                ckpt_path = uploaded_ckpt.get("name") or uploaded_ckpt.get("path")
            elif hasattr(uploaded_ckpt, "name"):
                ckpt_path = uploaded_ckpt.name
            else:
                ckpt_path = str(uploaded_ckpt)
        else:
            # Auto-search for best_model.pt / best_checkpoint.pt in local folders / Drive / Report
            for cand in ["/content/best_model.pt",
                         "/content/best_checkpoint.pt",
                         "/content/m10_checkpoints/checkpoints/best_checkpoint.pt",
                         "/content/drive/MyDrive/extracted_dataset_from_colab/Trained_Model_Weights/checkpoints/best_checkpoint.pt",
                         "/content/drive/MyDrive/extracted_dataset_from_colab/Trained_Model_Weights/checkpoints/best_model.pt",
                         os.path.join(current_dir, "best_model.pt"),
                         os.path.join(current_dir, "best_checkpoint.pt"),
                         os.path.join(current_dir, "..", "Report", "checkpoints", "best_model.pt")]:
                if os.path.exists(cand):
                    ckpt_path = cand
                    break

        ckpt_loaded = False
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                if "embedding_module" in state_dict:
                    embed_mod.load_state_dict(state_dict["embedding_module"], strict=False)
                if "edps_module" in state_dict:
                    edps_mod.load_state_dict(state_dict["edps_module"], strict=False)
                ckpt_loaded = True
            except Exception:
                pass

        embed_mod.eval()
        edps_mod.eval()

        with torch.no_grad():
            embeddings, embed_meta = embed_mod(patches, meta)
            edps_out = edps_mod(embeddings, stats, meta, embed_meta, adj)

        scores = edps_out.importance_scores.numpy()
        binary_mask = edps_out.binary_mask.numpy()
        n_total = len(scores)
        n_kept = int(binary_mask.sum())
        keep_ratio = (n_kept / n_total) * 100.0
        pruned_ratio = 100.0 - keep_ratio
        flops_reduction = 100.0 / max(1.0, keep_ratio)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        event_img = np.sum(np.abs(grid_np), axis=0)

        # Left: Baseline
        ax1.imshow(event_img, cmap="gray_r", origin="upper")
        ax1.set_title(f"Baseline Standard Transformer\n(100% Tokens = {n_total}/{n_total} Processed)", fontsize=11, fontweight="bold", color="crimson")
        for m in meta:
            rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.5, edgecolor="blue", facecolor="none", alpha=0.3)
            ax1.add_patch(rect)
        for b in window_boxes:
            bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            rect = patches_plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor="gold", facecolor="none")
            ax1.add_patch(rect)
        ax1.set_axis_off()

        # Right: Proposed EDPS Model
        ax2.imshow(event_img, cmap="gray_r", origin="upper")
        ax2.set_title(f"Proposed Dynamic Sparse Model (EDPS)\n(Kept {n_kept}/{n_total} Tokens [{keep_ratio:.1f}%] | Threshold={gate_thresh:.2f})", fontsize=11, fontweight="bold", color="darkgreen")

        for i, m in enumerate(meta):
            is_kept = binary_mask[i]
            score = scores[i]
            if is_kept:
                rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=1.5, edgecolor="lime", facecolor="green", alpha=0.25)
            else:
                rect = patches_plt.Rectangle((m.x0, m.y0), patch_size, patch_size, linewidth=0.5, edgecolor="red", facecolor="red", alpha=0.08)
            ax2.add_patch(rect)

            if show_scores:
                txt_color = "black" if is_kept else "darkred"
                ax2.text(m.x0 + 2, m.y0 + patch_size / 2, f"{score:.2f}", color=txt_color, fontsize=6, fontweight="bold")

        for b in window_boxes:
            bx, by, bw, bh = float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])
            rect = patches_plt.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor="gold", facecolor="none")
            ax2.add_patch(rect)
        ax2.set_axis_off()

        plt.tight_layout()

        rec_filename = os.path.basename(dat_path)
        ckpt_status = f"✅ Trained Model Loaded (`{os.path.basename(ckpt_path)}`)" if ckpt_loaded else "⚠️ Initial Untrained Weights"
        report_markdown = f"""
        ### ⚡ Performance & Efficiency Report
        - **Model Status:** {ckpt_status}
        - **Active File:** `{rec_filename}`
        - **Total Input Patches:** `{n_total}` patches
        - **Retained Tokens (Objects Focus):** `{n_kept}` (`{keep_ratio:.1f}%`) — **{pruned_ratio:.1f}% Pruned!**
        - **Energy / FLOPs Reduction:** **{flops_reduction:.2f}x Speedup**
        - **Active Window Events:** `{len(events['t']):,}` events
        - **Bounding Boxes Detected:** `{len(window_boxes)}` GT objects
        """
        return fig, report_markdown

    except Exception as e:
        err_msg = f"❌ Error processing demo:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, err_msg


def launch(share=True):
    recordings_map = get_available_recordings()
    rec_choices = list(recordings_map.keys())

    with gr.Blocks(title="EDPS Event Vision Demo") as demo:
        gr.Markdown("# ⚡ Energy-Efficient Dynamic Sparse Self-Attention for Event Vision")
        gr.Markdown("### Interactive Research Panel Presentation & Real-Time Token Pruning Demo")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### ⚡ Select File Uploaded to `/content/` (0s Load Time)")
                rec_dropdown = gr.Dropdown(choices=rec_choices, value=rec_choices[0] if rec_choices else None, label="Select Recording from /content/")
                
                gr.Markdown("#### 📁 Upload Custom Files (Optional)")
                dat_upload = gr.File(label="Upload .dat Event File")
                bbox_upload = gr.File(label="Upload _bbox.npy Label File")
                ckpt_upload = gr.File(label="🧠 Upload Trained Model Weights (best_checkpoint.pt)")

                gr.Markdown("#### 🎛️ Execution Controls")
                offset_slider = gr.Slider(minimum=0, maximum=10000, value=500, step=100, label="Window Time Offset (ms)")
                window_slider = gr.Slider(minimum=10, maximum=100, value=50, step=10, label="Window Duration (ms)")
                thresh_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.50, step=0.05, label="🎯 EDPS Gate Threshold")
                patch_size_dropdown = gr.Dropdown(choices=[16, 32], value=16, label="Patch Size (pixels)")
                show_scores_checkbox = gr.Checkbox(value=True, label="🔢 Show Numerical Scores (0.00 to 1.00) on Patches")
                btn = gr.Button("⚡ Run Token Selection Comparison", variant="primary")

            with gr.Column(scale=2):
                plot_output = gr.Plot(label="Real-Time Token Selection & Pruning Overlay")
                metrics_output = gr.Markdown()

        btn.click(
            fn=process_demo,
            inputs=[rec_dropdown, dat_upload, bbox_upload, ckpt_upload, offset_slider, window_slider, thresh_slider, patch_size_dropdown, show_scores_checkbox],
            outputs=[plot_output, metrics_output],
        )

    demo.launch(share=share, debug=False)


if __name__ == "__main__":
    launch(share=True)
