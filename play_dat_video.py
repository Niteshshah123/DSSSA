"""
================================================================================
Prophesee Gen1 DAT Event Video Player & MP4 Exporter
================================================================================
Plays any Prophesee .dat event file as a 2D event video frame-by-frame,
and optionally saves it as a standard .mp4 video file!

Usage (Play interactively):
    python play_dat_video.py --dat_path "/path/to/recording.dat"

Usage (Export to MP4 video):
    python play_dat_video.py --dat_path "/path/to/recording.dat" --save_mp4 "event_video.mp4"
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Ensure local project modules are importable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from event_parser import EventParser
from voxel_grid import VoxelGridConfig, events_to_voxel_grid


def play_or_export_dat(dat_path: str, bbox_path: str = None, save_mp4: str = None, draw_boxes: bool = True, fps: int = 30, window_ms: int = 33):
    if not os.path.exists(dat_path):
        print(f"❌ Error: File not found at {dat_path}")
        return

    # Auto-find bbox_path if not provided
    if bbox_path is None or not os.path.exists(bbox_path):
        possible_bbox = dat_path.replace("_td.dat", "_bbox.npy").replace(".dat", ".npy")
        if os.path.exists(possible_bbox):
            bbox_path = possible_bbox

    parser = EventParser()
    header = parser.reader.parse_header(dat_path)
    t_min, t_max, _ = parser.reader.get_time_range(dat_path, header)

    height = header["height"]
    width = header["width"]
    duration_sec = (t_max - t_min) / 1e6

    print("=" * 60)
    print("PROPHESEE DAT EVENT VIDEO PLAYER")
    print("=" * 60)
    print(f"File     : {os.path.basename(dat_path)}")
    print(f"BBox File: {os.path.basename(bbox_path) if bbox_path else 'None'}")
    print(f"Sensor   : {width} x {height} pixels")
    print(f"Duration : {duration_sec:.2f} seconds")
    print(f"Time Range: {t_min} us -> {t_max} us")
    print("=" * 60)

    # Prepare VideoWriter if saving MP4
    writer = None
    if save_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_mp4, fourcc, fps, (width, height))
        print(f"📹 Exporting video to: {save_mp4}")

    step_us = int((1.0 / fps) * 1e6)
    window_us = int(window_ms * 1000)

    t_curr = t_min
    frame_idx = 0

    print("Playing event stream...")

    while t_curr + window_us <= t_max:
        t_end = t_curr + window_us
        events = parser.load_events_window(dat_path, t_start=t_curr, t_end=t_end, validate=False)

        # Build 2D Frame (ON events = Green, OFF events = Red, Background = Dark)
        frame_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        if len(events["t"]) > 0:
            x = events["x"].astype(np.int64)
            y = events["y"].astype(np.int64)
            p = events["p"]

            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            x, y, p = x[valid], y[valid], p[valid]

            on_mask = p > 0
            off_mask = ~on_mask

            frame_rgb[y[on_mask], x[on_mask]] = [0, 255, 0]      # Green
            frame_rgb[y[off_mask], x[off_mask]] = [0, 0, 255]    # Red

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw Ground-Truth Bounding Boxes if requested
        if draw_boxes and bbox_path and os.path.exists(bbox_path):
            try:
                boxes = parser.load_annotations(bbox_path)
                if len(boxes) > 0 and hasattr(boxes, "dtype") and boxes.dtype.names is not None:
                    t_field = "t" if "t" in boxes.dtype.names else ("ts" if "ts" in boxes.dtype.names else boxes.dtype.names[0])
                    mask = (boxes[t_field] >= t_curr) & (boxes[t_field] < t_end)
                    for b in boxes[mask]:
                        bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
                        cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                        cls_id = int(b["class_id"]) if "class_id" in boxes.dtype.names else 0
                        cls_name = "Car" if cls_id == 0 else "Pedestrian"
                        cv2.putText(frame_bgr, cls_name, (bx, max(12, by - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            except Exception:
                pass

        # Overlay video info text
        current_sec = (t_curr - t_min) / 1e6
        cv2.putText(
            frame_bgr,
            f"Time: {current_sec:.2f}s / {duration_sec:.2f}s | Events: {len(events['t']):,}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if writer:
            writer.write(frame_bgr)

        # Show interactive video window only if GUI display is available
        if "DISPLAY" in os.environ and not save_mp4:
            try:
                cv2.imshow("Prophesee DAT Event Video Player (Press Q to quit)", frame_bgr)
                key = cv2.waitKey(int(1000 / fps)) & 0xFF
                if key == ord("q") or key == 27:
                    print("\nPlayback stopped by user.")
                    break
            except Exception:
                pass

        if frame_idx % (fps * 10) == 0 or frame_idx == 0:
            print(f"  [Progress] Frame {frame_idx} | Time: {current_sec:.1f}s / {duration_sec:.1f}s | Events: {len(events['t']):,}")

        t_curr += step_us
        frame_idx += 1

    if writer:
        writer.release()
        print(f"✅ Video export finished! Saved to `{save_mp4}`.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Play or export Prophesee .dat event files as 2D video.")
    ap.add_argument("--dat_path", type=str, required=True, help="Path to input .dat file")
    ap.add_argument("--bbox_path", type=str, default=None, help="Optional path to _bbox.npy file")
    ap.add_argument("--save_mp4", type=str, default=None, help="Optional output .mp4 video path")
    ap.add_argument("--draw_boxes", action="store_true", help="Draw yellow ground-truth bounding boxes")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    ap.add_argument("--window_ms", type=int, default=33, help="Event window accumulation duration in ms (default: 33)")
    args = ap.parse_args()

    play_or_export_dat(args.dat_path, bbox_path=args.bbox_path, save_mp4=args.save_mp4, draw_boxes=args.draw_boxes, fps=args.fps, window_ms=args.window_ms)
