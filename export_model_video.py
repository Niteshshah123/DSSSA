"""
================================================================================
Prophesee Gen1 DAT Model Inference MP4 Exporter (EDPS & Baseline)
================================================================================
Runs full end-to-end model inference (Patch Embedding -> EDPS Gating ->
Transformer Encoder -> Sparse Detection Head) frame-by-frame on any .dat event
file and exports an MP4 video showcasing live object detections and dynamic patch pruning.

Usage Examples (Run in Colab):

1. Export EDPS Proposed Model Detection Video:
    !python export_model_video.py \
        --dat_path "/content/17-10-18_18-24-24_61500000_121500000_td.dat" \
        --checkpoint "/content/checkpoints_edps_proposed/best_model.pt" \
        --selection_mode normal \
        --gate_thresh 0.50 \
        --conf_thresh 0.15 \
        --save_mp4 "/content/edps_model_detection_video.mp4"

2. Export Baseline Model Detection Video (100% Dense Tokens):
    !python export_model_video.py \
        --dat_path "/content/17-10-18_18-24-24_61500000_121500000_td.dat" \
        --checkpoint "/content/checkpoints_baseline/best_model.pt" \
        --selection_mode baseline_no_pruning \
        --conf_thresh 0.15 \
        --save_mp4 "/content/baseline_model_detection_video.mp4"
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch

# Ensure local project modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from event_parser import EventParser
from voxel_grid import VoxelGridConfig, events_to_voxel_grid
from patch_embedding import PatchConfig, PatchEmbeddingConfig, PatchEmbeddingModule, partition_voxel_grid, compute_patch_activity_stats, compute_patch_grid_dims, build_adjacency_map
from edps_module import EDPSConfig, EDPSModule, compute_gated_selected_embeddings, pad_token_sequences
from transformer_encoder import TransformerEncoderConfig, TransformerEncoder
from detection_head_config import DetectionHeadConfig
from sparse_detection_head import SparseDetectionHead, decode_predictions


CLASS_NAMES = ["Car", "Pedestrian", "Truck", "Vehicle"]
CLASS_COLORS_BGR = [
    (255, 212, 0),   # Cyan/Blue for Car
    (129, 185, 16),  # Emerald Green for Pedestrian
    (11, 158, 245),  # Amber/Orange for Truck
    (153, 72, 236),  # Pink/Purple for Vehicle
]


def load_model_components(ckpt_path, selection_mode="normal", gate_thresh=0.50, embedding_dim=256, num_bins=10, patch_size=16, n_rows=15, n_cols=19, device="cpu"):
    embed_cfg = PatchEmbeddingConfig(embedding_dim=embedding_dim)
    embed_mod = PatchEmbeddingModule(embed_cfg, num_bins, patch_size, n_rows, n_cols).to(device)

    edps_cfg = EDPSConfig(gate_threshold=gate_thresh, selection_mode=selection_mode)
    edps_mod = EDPSModule(edps_cfg, embedding_dim, num_bins).to(device)

    encoder_cfg = TransformerEncoderConfig(embedding_dim=embedding_dim, num_heads=4, num_layers=2)
    encoder = TransformerEncoder(encoder_cfg).to(device)

    head_cfg = DetectionHeadConfig(embedding_dim=embedding_dim, num_classes=2)
    detection_head = SparseDetectionHead(head_cfg).to(device)

    ckpt_status = "Default Untrained Weights"
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            try:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            except Exception:
                ckpt = torch.load(ckpt_path, map_location=device)
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
                ckpt_status = f"Successfully loaded checkpoint: {os.path.basename(ckpt_path)}"
        except Exception as e:
            ckpt_status = f"Warning loading checkpoint: {str(e)}"
    else:
        print(f"⚠️ Checkpoint not found at '{ckpt_path}'. Running with initialized weights.")

    embed_mod.eval(); edps_mod.eval(); encoder.eval(); detection_head.eval()
    return embed_mod, edps_mod, encoder, detection_head, head_cfg, ckpt_status


def export_model_video(
    dat_path: str,
    ckpt_path: str = None,
    save_mp4: str = "model_detection_video.mp4",
    selection_mode: str = "normal",
    gate_thresh: float = 0.50,
    conf_thresh: float = 0.15,
    patch_size: int = 16,
    fps: int = 30,
    window_ms: int = 33,
    duration_sec: float = 0.0,
    draw_patches: bool = True,
    device_name: str = None,
):
    if not os.path.exists(dat_path):
        print(f"❌ Error: .dat file not found at {dat_path}")
        return

    device = torch.device(device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu"))

    parser = EventParser()
    header = parser.reader.parse_header(dat_path)
    t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)

    height = header["height"]
    width = header["width"]
    total_duration_sec = (t_max - t_min) / 1e6

    print("=" * 65)
    print("PROPHESEE GEN1 MODEL INFERENCE MP4 EXPORTER")
    print("=" * 65)
    print(f"Event Stream : {os.path.basename(dat_path)}")
    print(f"Checkpoint   : {os.path.basename(ckpt_path) if ckpt_path else 'None'}")
    print(f"Selection    : Mode='{selection_mode}' | Gate Threshold={gate_thresh}")
    print(f"Score Cutoff : {conf_thresh}")
    print(f"Resolution   : {width} x {height} | FPS: {fps} | Device: {device.type.upper()}")
    print(f"Total Stream : {total_duration_sec:.2f} seconds")
    print("=" * 65)

    # Compute grid dimensions
    patch_cfg = PatchConfig(patch_size=patch_size, padding_mode="zero")
    n_rows, n_cols, eff_h, eff_w = compute_patch_grid_dims(height, width, patch_cfg)
    adj = build_adjacency_map(n_rows, n_cols, patch_cfg)

    # Load PyTorch model
    embed_mod, edps_mod, encoder, detection_head, head_cfg, load_msg = load_model_components(
        ckpt_path=ckpt_path,
        selection_mode=selection_mode,
        gate_thresh=gate_thresh,
        patch_size=patch_size,
        n_rows=n_rows,
        n_cols=n_cols,
        device=device,
    )
    print(f"ℹ️ {load_msg}")

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_mp4, fourcc, fps, (width, height))
    print(f"📹 Writing video stream to: `{save_mp4}`")

    step_us = int((1.0 / fps) * 1e6)
    window_us = int(window_ms * 1000)

    end_us_target = t_max
    if duration_sec > 0:
        end_us_target = min(t_max, t_min + int(duration_sec * 1e6))

    t_curr = t_min
    frame_idx = 0
    voxel_cfg = VoxelGridConfig(num_bins=10, temporal_mode="bilinear", polarity_encoding="signed")

    print("\nProcessing video frames...")

    while t_curr + window_us <= end_us_target:
        t_end = t_curr + window_us
        events = parser.load_events_window(dat_path, t_start=t_curr, t_end=t_end, validate=False)

        # 1. Render Base 2D Event Frame (ON = Green, OFF = Red)
        frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        if len(events["t"]) > 0:
            x = events["x"].astype(np.int64)
            y = events["y"].astype(np.int64)
            p = events["p"]

            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x, y, p = x[valid], y[valid], p[valid]

            on_mask = p > 0
            off_mask = ~on_mask

            frame_rgb[y[on_mask], x[on_mask]] = [0, 255, 0]      # Green (ON events)
            frame_rgb[y[off_mask], x[off_mask]] = [0, 0, 255]    # Red (OFF events)

            # Build 3D Voxel Grid for Model
            grid_np = events_to_voxel_grid(
                events["t"], events["x"], events["y"], events["p"],
                t_curr, t_end,
                height, width,
                voxel_cfg,
            )
            grid_tensor = torch.from_numpy(grid_np).to(device)

            patches, meta = partition_voxel_grid(grid_tensor, patch_cfg)
            stats = compute_patch_activity_stats(patches, voxel_cfg)

            # Run Model Inference
            with torch.no_grad():
                emb, em_meta = embed_mod(patches, meta)
                out_edps = edps_mod(emb, stats, meta, em_meta, adj)
                gated = compute_gated_selected_embeddings(out_edps)
                padded, mask = pad_token_sequences([gated])
                encoded = encoder(padded, attention_mask=mask)
                raw_dets = detection_head(encoded)
                dets = decode_predictions(raw_dets, out_edps.selected_metadata, head_cfg, attention_mask=mask, batch_index=0)

            n_total = len(out_edps.importance_scores)
            n_kept = int(out_edps.binary_mask.sum().item())
            pruned_pct = (1.0 - (n_kept / max(1, n_total))) * 100.0
            mask_np = out_edps.binary_mask.cpu().numpy()
        else:
            dets = []
            n_total = n_rows * n_cols
            n_kept = n_total
            pruned_pct = 0.0
            mask_np = np.ones(n_total, dtype=bool)

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 2. Draw Retained / Pruned Patch Overlays
        if draw_patches and len(meta) == len(mask_np):
            overlay = frame_bgr.copy()
            for idx, m in enumerate(meta):
                kept = mask_np[idx]
                if selection_mode == "normal":
                    if kept:
                        cv2.rectangle(overlay, (m.x0, m.y0), (m.x0 + patch_size, m.y0 + patch_size), (0, 255, 128), 1)
                else:
                    cv2.rectangle(overlay, (m.x0, m.y0), (m.x0 + patch_size, m.y0 + patch_size), (255, 100, 0), 1)
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, frame_bgr)

        # 3. Draw Model-Predicted Bounding Boxes
        shown_dets = [d for d in dets if d.combined_score >= conf_thresh]
        for d in shown_dets:
            color_bgr = CLASS_COLORS_BGR[d.predicted_class % len(CLASS_COLORS_BGR)]
            x0, y0 = int(d.box_x0), int(d.box_y0)
            x1, y1 = int(d.box_x1), int(d.box_y1)
            bw, bh = max(x1 - x0, 10), max(y1 - y0, 10)

            # Draw outer rectangle
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + bw, y0 + bh), color_bgr, 2)

            # Label box header
            cls_name = CLASS_NAMES[d.predicted_class % len(CLASS_NAMES)]
            label_str = f"{cls_name} {d.combined_score:.2f}"
            (w_lbl, h_lbl), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            y_text = max(y0 - 4, h_lbl + 2)
            cv2.rectangle(frame_bgr, (x0, y_text - h_lbl - 2), (x0 + w_lbl + 4, y_text + 2), (15, 23, 42), -1)
            cv2.rectangle(frame_bgr, (x0, y_text - h_lbl - 2), (x0 + w_lbl + 4, y_text + 2), color_bgr, 1)
            cv2.putText(frame_bgr, label_str, (x0 + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Top Information Banner
        current_sec = (t_curr - t_min) / 1e6
        banner_str = f"Time: {current_sec:.2f}s | Mode: {selection_mode} | Kept: {n_kept}/{n_total} ({pruned_pct:.1f}% Pruned) | Detections: {len(shown_dets)}"
        
        cv2.rectangle(frame_bgr, (0, 0), (width, 24), (15, 23, 42), -1)
        cv2.putText(frame_bgr, banner_str, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (241, 245, 249), 1, cv2.LINE_AA)

        writer.write(frame_bgr)

        if frame_idx % (fps * 5) == 0 or frame_idx == 0:
            print(f"  [Frame {frame_idx:4d}] Time: {current_sec:5.1f}s | Tokens: {n_kept:3d}/{n_total:3d} ({pruned_pct:4.1f}% pruned) | Detections: {len(shown_dets)}")

        t_curr += step_us
        frame_idx += 1

    writer.release()
    print("=" * 65)
    print(f"✅ Video Export Complete! Saved to: `{save_mp4}` ({frame_idx} total frames).")
    print("=" * 65)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export Model Predictions & Dynamic Pruning as MP4 Video.")
    ap.add_argument("--dat_path", type=str, required=True, help="Path to input .dat event file")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    ap.add_argument("--save_mp4", type=str, default="/content/model_detection_video.mp4", help="Output video MP4 path")
    ap.add_argument("--selection_mode", type=str, default="normal", choices=["normal", "baseline_no_pruning"], help="EDPS selection mode")
    ap.add_argument("--gate_thresh", type=float, default=0.50, help="EDPS gating threshold tau (default: 0.50)")
    ap.add_argument("--conf_thresh", type=float, default=0.15, help="Detection score cutoff (default: 0.15)")
    ap.add_argument("--patch_size", type=int, default=16, help="Patch size in pixels (default: 16)")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    ap.add_argument("--window_ms", type=int, default=33, help="Event window duration in ms (default: 33)")
    ap.add_argument("--duration_sec", type=float, default=0.0, help="Max video duration in seconds (0 = full stream)")
    ap.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda or cpu)")
    args = ap.parse_args()

    export_model_video(
        dat_path=args.dat_path,
        ckpt_path=args.checkpoint,
        save_mp4=args.save_mp4,
        selection_mode=args.selection_mode,
        gate_thresh=args.gate_thresh,
        conf_thresh=args.conf_thresh,
        patch_size=args.patch_size,
        fps=args.fps,
        window_ms=args.window_ms,
        duration_sec=args.duration_sec,
        device_name=args.device,
    )
