#!/usr/bin/env python3
"""
VGA Policy Batch Benchmark Runner for Headless Kaggle Cloud Execution.
Runs full pipeline: dependencies -> training -> closed-loop simulation -> artifacts.
Designed for 100% headless, non-interactive execution via Kaggle CLI (kaggle kernels push).
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

def run_cmd(cmd: str, desc: str, check: bool = True):
    print(f"\n{'='*70}")
    print(f"▶ {desc}")
    print(f"{'='*70}")
    print(f"Executing: {cmd}\n")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, check=check)
    dt = time.time() - t0
    print(f"\n✔ Completed in {dt:.1f}s (Exit code: {res.returncode})")
    return res

def main():
    print("=================================================================")
    print("   VGA PoC: Headless Kaggle CLI Cloud Batch Execution Runner     ")
    print("=================================================================")

    # 1. Environment & Hardware Diagnostics
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"

    run_cmd("nvidia-smi", "NVIDIA GPU Diagnostics", check=False)

    # 2. Install Headless EGL System Libraries (Ubuntu 22.04 & 24.04 compatible)
    apt_cmd = (
        "apt-get update -qq && "
        "apt-get install -y -qq libgl1 libglx-mesa0 libosmesa6-dev || "
        "apt-get install -y -qq libgl1-mesa-glx libosmesa6-dev || true"
    )
    run_cmd(apt_cmd, "Installing Headless EGL System Drivers", check=False)

    # 3. Install Python Dependencies
    pip_cmd = (
        'pip install --quiet --break-system-packages "av<14" datasets num2words '
        'git+https://github.com/huggingface/lerobot.git '
        'zarr einops scipy torchvision peft accelerate matplotlib imageio imageio-ffmpeg "hf-libero>=0.1.4" mujoco 2>/dev/null || '
        'pip install --quiet "av<14" datasets num2words '
        'git+https://github.com/huggingface/lerobot.git '
        'zarr einops scipy torchvision peft accelerate matplotlib imageio imageio-ffmpeg "hf-libero>=0.1.4" mujoco'
    )
    run_cmd(pip_cmd, "Installing Python Robotics Dependencies (LeRobot, LIBERO, MuJoCo)")

    # 4. Clone / Sync latest repository from GitHub
    repo_dir = "/kaggle/working/vga-poc"
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone https://github.com/Arkz-Deepak/vga-poc.git {repo_dir}", "Cloning latest vga-poc repository")
    else:
        run_cmd(f"cd {repo_dir} && git fetch origin && git reset --hard origin/main", "Updating vga-poc to latest origin/main")

    # 5. Output directories
    ckpt_dir = "/kaggle/working/checkpoints"
    results_dir = "/kaggle/working/results"
    video_dir = "/kaggle/working/results/videos"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # 6. Execute 1,500-Step Training with In-Memory Caching (~60s on Tesla T4)
    train_cmd = (
        f"python {repo_dir}/scripts/train_vga.py "
        f"--task_id 0 "
        f"--shots 20 "
        f"--steps 1500 "
        f"--batch_size 16 "
        f"--lr 0.0002 "
        f"--in_memory "
        f"--output_dir {ckpt_dir}"
    )
    run_cmd(train_cmd, "High-Throughput Policy Training (1,500 steps, 20 shots, in-memory RAM caching)")

    # 7. Execute Closed-Loop MuJoCo Simulation (Pure Neural Policy)
    ckpt_path = os.path.join(ckpt_dir, "vga_libero_policy.pt")
    if not os.path.exists(ckpt_path):
        fallback = os.path.join(ckpt_dir, "vga_libero_20shot.pt")
        ckpt_path = fallback if os.path.exists(fallback) else None

    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}!")

    eval_cmd = (
        f"python {repo_dir}/scripts/eval_mujoco_closed_loop.py "
        f"--checkpoint {ckpt_path} "
        f"--task_id 0 "
        f"--num_episodes 5 "
        f"--max_steps 280 "
        f"--pure_policy "
        f"--friction_boost 1.0 "
        f"--flip_image "
        f"--video_dir {video_dir} "
        f"--output_json {results_dir}/closed_loop_simulation_results.json"
    )
    run_cmd(eval_cmd, "Closed-Loop MuJoCo Simulation Benchmark (Pure Policy, Standard Physics)")

    # 8. Summary of VGA Results
    res_json = os.path.join(results_dir, "closed_loop_simulation_results.json")
    if os.path.exists(res_json):
        with open(res_json, "r") as f:
            data = json.load(f)
        print("\n" + "="*70)
        print("          OFFICIAL VGA CLOSED-LOOP BENCHMARK RESULTS")
        print("="*70)
        print(f"Overall Success Rate: {data.get('overall_success_rate_pct', 0.0):.1f}%")
        print(f"Mean RMS Jerk:        {data.get('rms_jerk_mps3', 0.0):.2f} m/s³")
        print("="*70 + "\n")

    # 9. Benchmark SmolVLA-450M Baseline
    smol_video_dir = "/kaggle/working/results/videos_smolvla"
    os.makedirs(smol_video_dir, exist_ok=True)
    smol_cmd = (
        f"python {repo_dir}/scripts/eval_smolvla_libero.py "
        f"--policy_path lerobot/smolvla_libero "
        f"--task_id 0 "
        f"--num_episodes 5 "
        f"--n_action_steps 10 "
        f"--video_dir {smol_video_dir} "
        f"--output_json {results_dir}/smolvla_simulation_results.json"
    )
    run_cmd(smol_cmd, "SmolVLA-450M Baseline Evaluation", check=False)

    # 10. Composite Side-by-Side Comparison Video (VGA vs SmolVLA)
    import glob
    vga_vids = sorted(glob.glob(f"{video_dir}/*success*.mp4") or glob.glob(f"{video_dir}/*.mp4"))
    smol_vids = sorted(glob.glob(f"{smol_video_dir}/*SUCC*.mp4") or glob.glob(f"{smol_video_dir}/*.mp4"))
    comp_out = f"{results_dir}/comparison_vga_vs_smolvla_task0.mp4"
    if vga_vids and smol_vids:
        grid_cmd = (
            f"python {repo_dir}/scripts/create_comparison_grid.py "
            f"--videos {vga_vids[0]} {smol_vids[0]} "
            f"--labels 'VGA (Few-Shot Policy)' 'SmolVLA-450M Baseline' "
            f"--task_desc 'pick up the black bowl between the plate and the ramekin and place it on the plate' "
            f"--task_id 0 "
            f"--output {comp_out}"
        )
        run_cmd(grid_cmd, "Generating Side-by-Side Comparison Video Grid (VGA vs SmolVLA)", check=False)

    # List generated artifacts
    print("\nSaved Artifacts in /kaggle/working/:")
    run_cmd("ls -lh /kaggle/working/checkpoints /kaggle/working/results /kaggle/working/results/videos /kaggle/working/results/videos_smolvla", "Listing Output Files", check=False)
    print("\n🎉 Headless batch job completed successfully!")

if __name__ == "__main__":
    main()
