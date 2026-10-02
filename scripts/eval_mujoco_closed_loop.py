"""
Closed-Loop MuJoCo Simulation Benchmark for VGA Policy on LIBERO-Spatial.

Runs live, interactive physical simulation rollouts using MuJoCo and the official
LIBERO benchmark environment via LeRobot (`lerobot.envs.libero`).

Evaluation Workflow:
1. Spawns the MuJoCo / RoboSuite simulation environment for the 3 target tasks.
2. In each timestep, the simulated Panda robot's cameras provide live observations.
3. The VGA policy runs real-time inference (using action chunking & queueing).
4. Physical actions are executed by the MuJoCo physics engine step-by-step.
5. Verifies physical task completion (e.g. bowl placed on plate, can inside basket).
6. Computes exact Success Rate (SR) and records MP4 video rollouts.
"""

import argparse
import builtins
import json
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional

# Automatically respond 'n' to any interactive prompts (such as LIBERO's initial setup prompt)
builtins.input = lambda *args, **kwargs: "n"

import numpy as np
import torch

# Ensure project root is in sys.path
root_dir = pathlib.Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Configure headless rendering for Kaggle / Cloud environments
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Pre-populate ~/.libero/config.yaml dynamically with correct path mappings
libero_dir = pathlib.Path.home() / ".libero"
libero_dir.mkdir(parents=True, exist_ok=True)
config_yaml = libero_dir / "config.yaml"
try:
    import libero
    lib_root = pathlib.Path(libero.__file__).resolve().parent

    # Candidate locations for init_states (.pruned_init files)
    init_dir = lib_root / "init_files"
    init_candidates = [
        lib_root / "init_files",
        lib_root / "libero" / "init_files",
        lib_root / "init_states",
        lib_root / "libero" / "init_states",
        pathlib.Path.home() / ".cache" / "libero" / "assets" / "init_files",
        pathlib.Path.home() / ".cache" / "libero" / "init_files",
        pathlib.Path.home() / ".cache" / "libero" / "assets",
        pathlib.Path.home() / ".cache" / "libero",
        pathlib.Path("/root/.cache/libero/assets/init_files"),
        pathlib.Path("/root/.cache/libero/init_files"),
        pathlib.Path("/root/.cache/libero/assets"),
        pathlib.Path("/root/.cache/libero"),
    ]
    for cand in init_candidates:
        if cand.is_dir():
            # Check if it contains .pruned_init files directly or in subfolders
            pruned_files = list(cand.rglob("*.pruned_init"))
            if pruned_files:
                first_file = pruned_files[0]
                # LeRobot calls root / task.problem_folder / filename.name
                # So if first_file is .../libero_spatial/task.pruned_init, root is its parent's parent
                if first_file.parent.name in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
                    init_dir = first_file.parent.parent
                else:
                    init_dir = first_file.parent
                break
            elif not init_dir.is_dir():
                init_dir = cand

    bddl_dir = lib_root / "bddl_files"
    for cand in [lib_root / "bddl_files", lib_root / "libero" / "bddl_files"]:
        if cand.is_dir():
            bddl_dir = cand
            break

    assets_dir = lib_root / "assets"
    for cand in [
        lib_root / "assets",
        lib_root / "libero" / "assets",
        pathlib.Path.home() / ".cache" / "libero" / "assets",
        pathlib.Path("/root/.cache/libero/assets"),
        pathlib.Path.home() / ".cache" / "libero",
        pathlib.Path("/root/.cache/libero"),
    ]:
        if cand.is_dir():
            assets_dir = cand
            break

    config_yaml.write_text(
        f"benchmark_root: '{str(lib_root)}'\n"
        f"datasets: '{str(lib_root / 'datasets')}'\n"
        f"bddl_files: '{str(bddl_dir)}'\n"
        f"init_states: '{str(init_dir)}'\n"
        f"assets: '{str(assets_dir)}'\n"
    )
except Exception:
    pass

from configs.poc_config import ModelConfig, cfg

# Allow ModelConfig in PyTorch 2.6+ safe globals
try:
    import torch.serialization
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([ModelConfig])
except Exception:
    pass

from data.dataset import Normalizer
from models.vga_policy import VGAPolicy


def resolve_file(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    if os.path.isfile(path_str):
        return path_str
    basename = os.path.basename(path_str)
    search_dirs = [
        ".",
        "checkpoints",
        "configs",
        str(root_dir),
        os.path.join(str(root_dir), "checkpoints"),
        os.path.join(str(root_dir), "configs"),
        "/kaggle/working",
        "/kaggle/working/checkpoints",
        "/kaggle/working/vga-poc",
        "/kaggle/working/vga-poc/checkpoints",
        "/kaggle/working/vga-poc/configs",
    ]
    for d in search_dirs:
        candidate = os.path.join(d, basename)
        if os.path.isfile(candidate):
            return candidate
        candidate_rel = os.path.join(d, path_str)
        if os.path.isfile(candidate_rel):
            return candidate_rel
    return None


def run_closed_loop_evaluation(
    checkpoint_path: str,
    stats_path: str,
    target_tasks: List[str],
    num_episodes_per_task: int = 5,
    max_steps_per_episode: int = 280,
    device: str = "cuda",
    record_videos: bool = True,
    video_dir: str = "results/videos",
    output_json: str = "results/closed_loop_simulation_results.json",
    schmitt_low: float = 0.45,
    schmitt_high: float = 0.55,
):
    print("=================================================================")
    print("   VGA Closed-Loop MuJoCo Simulation Benchmark (LIBERO-Spatial)   ")
    print("=================================================================")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    # 1. Check for LIBERO simulation dependencies
    try:
        from lerobot.envs.libero import LiberoEnv, _get_suite
    except ImportError as e:
        print("\n❌ ERROR: LIBERO simulation environment is not installed!")
        print("To run live MuJoCo simulation on Kaggle, install dependencies with:")
        print("  !apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libosmesa6-dev")
        print("  !pip install -q 'hf-libero>=0.1.4' imageio[ffmpeg]")
        print(f"Detailed import error: {e}")
        return

    try:
        import imageio
    except ImportError:
        record_videos = False
        print("Notice: imageio not installed; video recording disabled.")

    # 2. Instantiate and Load Policy
    normalizer = Normalizer(stats_path=stats_path)
    policy = VGAPolicy(
        normalizer=normalizer,
        vis_dim=cfg.vis_dim,
        lm_dim=cfg.lm_dim,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        num_visual_tokens=cfg.num_visual_tokens,
        num_lm_layers=12,
        action_dim=cfg.action_dim,
        action_horizon=cfg.action_horizon,
        prefix_len=cfg.prefix_len,
        dit_hidden_dim=384,
        dit_layers=cfg.dit_layers,
        euler_steps=cfg.euler_steps,
        beta_jerk=cfg.beta_jerk,
        w_rot=cfg.w_rot,
        schmitt_low=schmitt_low,
        schmitt_high=schmitt_high,
    ).to(device)

    # Load weights
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    if "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "policy" in ckpt:
        state_dict = ckpt["policy"]
    else:
        state_dict = ckpt

    cleaned_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    policy.load_state_dict(cleaned_state, strict=False)
    policy.eval()

    tok = policy.encoder.get_tokenizer()
    if tok is not None and getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token

    # 3. Instantiate Benchmark Suite
    print("\n--- Initializing LIBERO-Spatial Suite ---")
    suite = _get_suite("libero_spatial")
    total_suite_tasks = len(suite.tasks)
    print(f"LIBERO-Spatial Suite loaded: {total_suite_tasks} tasks available.")

    # Find task indices matching our exact target tasks
    matched_task_indices = []
    for target in target_tasks:
        target_clean = target.lower().replace(" ", "_")
        matched = False
        for idx in range(total_suite_tasks):
            t_name = suite.get_task(idx).name.lower()
            t_lang = suite.get_task(idx).language.lower()
            if target_clean == t_name or target_clean in t_name or target.lower() == t_lang:
                matched_task_indices.append((idx, suite.get_task(idx).language))
                matched = True
                break
        if not matched:
            print(f"Notice: Could not match exact task '{target}' in suite.")

    if not matched_task_indices:
        print("Warning: Could not match specific task names, using first 3 tasks in suite.")
        matched_task_indices = [(i, suite.get_task(i).language) for i in range(min(3, total_suite_tasks))]

    print(f"Target tasks to evaluate ({len(matched_task_indices)}):")
    for t_idx, t_name in matched_task_indices:
        print(f"  - Suite Task [{t_idx}]: {t_name}")

    # 4. Run Closed-Loop Rollouts
    if record_videos:
        os.makedirs(video_dir, exist_ok=True)
        print(f"Rollout videos will be saved to: {os.path.abspath(video_dir)}")

    task_results = {}
    total_successes = 0
    total_rollouts = 0

    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    for task_id, task_desc in matched_task_indices:
        clean_task_name = task_desc.replace(" ", "_")[:40]
        print(f"\n=======================================================")
        print(f"Task [{task_id}]: {task_desc}")
        print(f"=======================================================")

        try:
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name="libero_spatial",
                observation_width=cfg.img_size,
                observation_height=cfg.img_size,
                control_mode="relative",
                episode_length=max_steps_per_episode,
                init_states=True,
            )
            print(f"Initialized LiberoEnv with benchmark demonstration init_states=True.")
        except Exception as e_init:
            print(f"Notice: Loading fixed init_states failed ({e_init}). Initializing LiberoEnv with procedural BDDL reset (init_states=False)...")
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name="libero_spatial",
                observation_width=cfg.img_size,
                observation_height=cfg.img_size,
                control_mode="relative",
                episode_length=max_steps_per_episode,
                init_states=False,
            )

        # Pre-tokenize task instruction
        if tok is not None:
            tokenized = tok(
                task_desc,
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
        else:
            input_ids = torch.zeros((1, 48), dtype=torch.long, device=device)
            attention_mask = torch.ones((1, 48), dtype=torch.bool, device=device)

        task_successes = 0
        episode_lengths = []

        for ep in range(num_episodes_per_task):
            policy.reset()
            obs, info = env.reset(seed=ep + 100)  # Use fixed seed for reproducibility
            video_frames = []
            success = False
            prev_grip = None
            closed_steps = 0

            for step in range(max_steps_per_episode):
                # Format visual observation [H, W, 3] -> [1, 3, H, W]
                # LeRobot returns dict with "pixels": {"image": ..., "image2": ...}
                pixels = obs["pixels"]
                img_front_np = pixels.get("image", next(iter(pixels.values())))

                # Flip 180° to align with LeRobot / HuggingFace LIBERO convention (dims H and W)
                img_front_np = np.ascontiguousarray(img_front_np[::-1, ::-1])

                if record_videos and step % 2 == 0:
                    video_frames.append(img_front_np)

                # Convert to PyTorch float tensor in [-1, 1]
                img_tensor = torch.from_numpy(img_front_np).permute(2, 0, 1).float().unsqueeze(0).to(device)
                img_tensor = (img_tensor / 255.0) * 2.0 - 1.0

                batch = {
                    "image_front": img_tensor,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                # Real-time policy action query (uses 16-step chunk queue)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        action_tensor = policy.select_action(batch)
                else:
                    action_tensor = policy.select_action(batch)

                action_np = action_tensor.cpu().numpy()
                action_np = np.clip(action_np, -1.0, 1.0)

                # Monitor gripper state changes
                curr_grip = float(action_np[6])
                if curr_grip > 0:
                    closed_steps += 1
                if prev_grip is None or (curr_grip > 0 and prev_grip <= 0) or (curr_grip <= 0 and prev_grip > 0):
                    grip_name = "CLOSED (+1.0)" if curr_grip > 0 else "OPEN (-1.0)"
                    print(f"    [Step {step:3d}] Gripper state -> {grip_name}")
                prev_grip = curr_grip

                # Step MuJoCo physics engine
                obs, reward, terminated, truncated, info = env.step(action_np)

                if info.get("is_success", False) or reward > 0:
                    success = True
                    episode_lengths.append(step + 1)
                    print(f"  - Episode {ep + 1}/{num_episodes_per_task}: ✅ SUCCESS at step {step + 1}!")
                    break

                if terminated or truncated:
                    break

            if not success:
                episode_lengths.append(max_steps_per_episode)
                print(f"  - Episode {ep + 1}/{num_episodes_per_task}: ❌ Failed (timeout {max_steps_per_episode} steps, gripper closed: {closed_steps} steps)")

            if success:
                task_successes += 1

            # Save video replay of the rollout
            if record_videos and video_frames:
                status_str = "success" if success else "failed"
                video_filename = os.path.join(video_dir, f"task_{task_id}_ep_{ep}_{status_str}.mp4")
                try:
                    imageio.mimsave(video_filename, video_frames, fps=20)
                except Exception as ve:
                    pass

            total_rollouts += 1

        total_successes += task_successes
        task_sr = (task_successes / num_episodes_per_task) * 100.0
        avg_steps = float(np.mean(episode_lengths))

        task_results[task_desc] = {
            "success_rate_pct": task_sr,
            "successes": task_successes,
            "total_episodes": num_episodes_per_task,
            "avg_steps_to_finish": avg_steps,
        }
        print(f"Task Success Rate: {task_sr:.1f}% ({task_successes}/{num_episodes_per_task}) | Avg Steps: {avg_steps:.1f}")

    # 5. Print Overall Summary Table
    overall_sr = (total_successes / max(1, total_rollouts)) * 100.0
    print("\n" + "=" * 75)
    print("      LIBERO-SPATIAL CLOSED-LOOP SIMULATION RESULTS")
    print("=" * 75)
    print(f"{'Task Description':<50} | {'Success Rate':<12} | {'Avg Steps':<10}")
    print("-" * 75)
    for t_desc, r in task_results.items():
        print(f"{t_desc[:48]:<50} | {r['success_rate_pct']:>9.1f}% | {r['avg_steps_to_finish']:>9.1f}")
    print("-" * 75)
    print(f"{'OVERALL AVERAGE SUCCESS RATE':<50} | {overall_sr:>9.1f}% | {total_successes}/{total_rollouts}")
    print("=" * 75)

    final_results = {
        "overall_success_rate_pct": overall_sr,
        "total_successes": total_successes,
        "total_rollouts": total_rollouts,
        "tasks": task_results,
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✅ Closed-loop simulation metrics exported to: {output_json}")
    if record_videos:
        print(f"🎥 Simulation video replays saved to: {video_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VGA Policy in Closed-Loop MuJoCo Simulation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vga_libero_10shot.pt",
                        help="Path to trained VGA checkpoint")
    parser.add_argument("--stats_path", type=str, default="configs/action_stats.json",
                        help="Path to action normalization stats")
    parser.add_argument("--num_episodes", "--episodes", dest="num_episodes", type=int, default=5,
                        help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=280,
                        help="Maximum simulation steps per episode")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Target device")
    parser.add_argument("--record_videos", action="store_true", default=True,
                        help="Whether to save MP4 video replays")
    parser.add_argument("--video_dir", type=str, default="results/videos",
                        help="Directory to save MP4 videos")
    parser.add_argument("--schmitt_high", type=float, default=0.55,
                        help="Schmitt trigger gripper close threshold (default: 0.55)")
    parser.add_argument("--schmitt_low", type=float, default=0.45,
                        help="Schmitt trigger gripper open threshold (default: 0.45)")
    parser.add_argument("--output_json", type=str, default="results/closed_loop_simulation_results.json",
                        help="Path to save simulation metrics JSON")
    args = parser.parse_args()

    ckpt_resolved = resolve_file(args.checkpoint) or "checkpoints/vga_libero_10shot.pt"
    stats_resolved = resolve_file(args.stats_path) or "configs/action_stats.json"

    run_closed_loop_evaluation(
        checkpoint_path=ckpt_resolved,
        stats_path=stats_resolved,
        target_tasks=cfg.benchmark_tasks,
        num_episodes_per_task=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        device=args.device,
        record_videos=args.record_videos,
        video_dir=args.video_dir,
        output_json=args.output_json,
        schmitt_low=args.schmitt_low,
        schmitt_high=args.schmitt_high,
    )


if __name__ == "__main__":
    main()
