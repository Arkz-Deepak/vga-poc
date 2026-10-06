#!/usr/bin/env python3
"""
Composite Multi-Model or Multi-Variant LIBERO-Spatial Rollouts into a
Unified Academic Research Comparison Video (1x2 or 2x3 Grid).

Matches the exact aesthetic layout from the official VGA benchmark report:
  - Charcoal top header (#0d0f0f) with task instruction and benchmark subtitle
  - Individual panel subheaders (#171919) with variant / model names
  - Synchronized real-time (20 Hz) side-by-side or 2x3 grid playback
  - Automatic frame padding so finished rollouts hold their final (SUCCESS) state

Usage examples:
  # Side-by-side comparison: VGA vs SmolVLA
  python scripts/create_comparison_grid.py \
      --videos results/videos/task_0_ep_0_success.mp4 results/videos_smolvla/smolvla_task0_ep1_SUCC.mp4 \
      --labels "VGA (10-Shot Policy)" "SmolVLA-450M Baseline" \
      --task_desc "Pick up the black bowl between the plate and the ramekin and place it on the plate." \
      --task_id 0 \
      --output results/comparison_vga_vs_smolvla_task0.mp4

  # Full 6-variant ablation grid (2 rows x 3 cols):
  python scripts/create_comparison_grid.py \
      --videos v0.mp4 v1.mp4 v2.mp4 v3.mp4 v4.mp4 v5.mp4 \
      --labels "Baseline" "+ depth supervision" "+ camera rays" "+ smooth chunk joins" "All three add-ons" "Baseline + moving-avg filter" \
      --task_desc "Pick up the black bowl next to the cookie box and place it on the plate." \
      --task_id 6 \
      --grid_shape 2 3 \
      --output results/ablation_grid_task6.mp4
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Create comparison research grid video.")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files to tile together.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each video panel.")
    parser.add_argument("--task_desc", type=str, default="pick up the black bowl between the plate and the ramekin and place it on the plate", help="Language instruction for task.")
    parser.add_argument("--task_id", type=int, default=0, help="LIBERO-Spatial task ID.")
    parser.add_argument("--ep_idx", type=int, default=0, help="Starting state / episode index.")
    parser.add_argument("--shots", type=int, default=10, help="Demonstration budget.")
    parser.add_argument("--grid_shape", nargs=2, type=int, default=None, help="Grid dimensions: rows cols (e.g. 1 2 or 2 3).")
    parser.add_argument("--output", type=str, default="results/comparison_grid.mp4", help="Output MP4 file path.")
    parser.add_argument("--fps", type=int, default=20, help="Playback frame rate (default 20 Hz).")
    parser.add_argument("--panel_width", type=int, default=336, help="Width per pane in pixels.")
    return parser.parse_args()


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Reads all frames from an MP4 file as RGB numpy arrays."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # OpenCV reads BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def create_comparison_grid(
    video_paths: List[str],
    labels: Optional[List[str]],
    task_desc: str,
    task_id: int,
    ep_idx: int,
    shots: int,
    output_path: str,
    grid_shape: Optional[Tuple[int, int]] = None,
    fps: int = 20,
    panel_width: int = 336,
):
    num_videos = len(video_paths)
    if labels is None or len(labels) < num_videos:
        labels = [f"Variant {i + 1}" for i in range(num_videos)]

    # Determine grid shape
    if grid_shape is not None:
        nrows, ncols = grid_shape
    elif num_videos <= 2:
        nrows, ncols = 1, num_videos
    elif num_videos <= 4:
        nrows, ncols = 2, 2
    elif num_videos <= 6:
        nrows, ncols = 2, 3
    else:
        ncols = 3
        nrows = (num_videos + ncols - 1) // ncols

    print(f"Loading {num_videos} videos into a {nrows}x{ncols} grid...")
    all_video_frames = []
    max_frames = 0
    for v_path in video_paths:
        frames = load_video_frames(v_path)
        print(f"  - Loaded {len(frames)} frames from: {v_path}")
        all_video_frames.append(frames)
        if len(frames) > max_frames:
            max_frames = len(frames)

    if max_frames == 0:
        raise ValueError("No frames found across specified videos!")

    # Pad shorter videos with their final frame so success states remain visible
    for i in range(num_videos):
        if len(all_video_frames[i]) < max_frames:
            last_frame = all_video_frames[i][-1] if all_video_frames[i] else np.zeros((panel_width, panel_width, 3), dtype=np.uint8)
            pad_count = max_frames - len(all_video_frames[i])
            all_video_frames[i].extend([last_frame] * pad_count)

    # Frame sizing
    total_w = ncols * panel_width
    header_h = 60
    subbar_h = 28

    # Derive sample frame aspect ratio
    sample_f = all_video_frames[0][0]
    sample_h, sample_w, _ = sample_f.shape
    aspect = sample_h / sample_w
    panel_h = int(panel_width * aspect)

    total_h = header_h + nrows * (subbar_h + panel_h)
    print(f"Output Video Dimensions: {total_w}x{total_h} @ {fps} fps ({max_frames} frames)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, total_h))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Clean text description
    clean_desc = task_desc.strip()
    if clean_desc.endswith("."):
        clean_desc = clean_desc[:-1]

    sub_meta = f"LIBERO-Spatial task {task_id}, start state {ep_idx} · {shots} demos/task, seed 0 · real time ({fps} Hz) · inset: wrist camera"

    for f_idx in range(max_frames):
        canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # 1. Overarching Top Header (#0d0f0f)
        canvas[:header_h, :] = (13, 15, 15)
        cv2.putText(canvas, clean_desc.lower(), (14, 26), font, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub_meta, (14, 48), font, 0.38, (175, 180, 180), 1, cv2.LINE_AA)

        # 2. Render each row and column
        for r in range(nrows):
            row_y_sub = header_h + r * (subbar_h + panel_h)
            row_y_vid = row_y_sub + subbar_h

            # Sub-header bar across entire row
            canvas[row_y_sub:row_y_vid, :] = (23, 25, 25)

            for c in range(ncols):
                idx = r * ncols + c
                col_x = c * panel_width

                if idx < num_videos:
                    label = labels[idx]
                    cv2.putText(canvas, label, (col_x + 10, row_y_sub + 19), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

                    frame_raw = all_video_frames[idx][f_idx]
                    frame_resized = cv2.resize(frame_raw, (panel_width, panel_h), interpolation=cv2.INTER_AREA)
                    canvas[row_y_vid:row_y_vid + panel_h, col_x:col_x + panel_width] = frame_resized

        # Convert RGB back to BGR for VideoWriter
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        out.write(canvas_bgr)

    out.release()
    print(f"✅ Comparison grid video generated successfully: {output_path}")


def main():
    args = parse_args()
    create_comparison_grid(
        video_paths=args.videos,
        labels=args.labels,
        task_desc=args.task_desc,
        task_id=args.task_id,
        ep_idx=args.ep_idx,
        shots=args.shots,
        output_path=args.output,
        grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
        fps=args.fps,
        panel_width=args.panel_width,
    )


if __name__ == "__main__":
    main()
