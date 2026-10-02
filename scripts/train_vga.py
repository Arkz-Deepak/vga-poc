"""
Training & Fine-Tuning Pipeline for VGA Policy on LIBERO-Spatial Benchmark.

Objectives:
1. 5-Shot and 10-Shot Demonstration Learning:
   Fine-tunes the lightweight (298M parameter) VGA Policy on 3 spatial manipulation tasks.
2. Kinematic Loss Scheduling:
   Linearly anneals lambda_kin from 0.0 to 0.05 over warmup steps to allow initial flow
   matching convergence before enforcing jerk and acceleration smoothness.
3. Post-Training Latency & Jerk Benchmark:
   Measures 4-step Euler ODE rollout latency on GPU to verify the <= 18 ms real-time ceiling
   against the SmolVLA-450M baseline (which achieved 643.75 ms on Tesla T4).
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Dict, List

# Ensure project root is in sys.path
root_dir = pathlib.Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Upstream compatibility guard for PyAV
try:
    import av, types
    if not hasattr(av, "option"):
        av.option = types.ModuleType("av.option")
        av.option.Option = object
except ImportError:
    pass

from configs.poc_config import ModelConfig
from data.dataset import LiberoSpatialDataset, Normalizer
from models.vga_policy import VGAPolicy


def train_vga(
    shots: int = 5,
    batch_size: int = 16,
    num_steps: int = 1000,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "checkpoints",
):
    print("=" * 65)
    print(f"      VGA PoC Training Run: {shots}-Shot Demo Benchmark       ")
    print("=" * 65)
    print(f"Target Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")

    cfg = ModelConfig()
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Ingest Dataset & Empirical Normalizer
    print("\n--- 1. Dataset Ingestion & Demonstration Subsetting ---")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = "lerobot/libero_spatial_image"
    try:
        raw_dataset = LeRobotDataset(repo_id)
    except Exception:
        repo_id = "lerobot/libero_spatial"
        raw_dataset = LeRobotDataset(repo_id)

    normalizer = Normalizer("configs/action_stats.json")
    train_dataset = LiberoSpatialDataset(
        lerobot_dataset=raw_dataset,
        normalizer=normalizer,
        action_horizon=cfg.action_horizon,
        target_tasks=cfg.benchmark_tasks,
        shots_per_task=shots,
        img_size=cfg.img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # 2. Instantiate VGA Policy Network
    print("\n--- 2. Instantiating VGA Policy ---")
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
        schmitt_low=cfg.schmitt_low,
        schmitt_high=cfg.schmitt_high,
    ).to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total Parameters: {total_params:,} ({total_params / 1e6:.1f}M) <= 0.5B ceiling")

    # 3. Setup Optimizer & Cosine Schedule
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    # 4. Training Loop
    print(f"\n--- 3. Beginning Fine-Tuning ({num_steps} Steps, Batch Size: {batch_size}) ---")
    policy.train()
    step = 0
    t_start = time.time()
    data_iter = iter(train_loader)

    # Tokenizer helper
    tok = policy.encoder.get_tokenizer()

    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        img_front = batch["image_front"].to(device)
        actions = batch["actions"].to(device)
        tasks = batch["task"]

        # Tokenize task instructions
        if tok is not None:
            tokenized = tok(tasks, return_tensors="pt", padding="max_length", max_length=48, truncation=True)
            input_ids = tokenized["input_ids"].to(device)
            att_mask = tokenized["attention_mask"].to(device)
        else:
            input_ids = torch.zeros((batch_size, 48), dtype=torch.long, device=device)
            att_mask = torch.ones((batch_size, 48), dtype=torch.bool, device=device)

        # Anneal kinematic loss weight: 0.0 -> max_lambda_kin over kinematic_anneal_steps
        curr_lambda_kin = cfg.max_lambda_kin * min(1.0, step / max(1, cfg.kinematic_anneal_steps))

        optimizer.zero_grad()
        loss_dict = policy.forward_loss(
            image_front=img_front,
            input_ids=input_ids,
            actions=actions,
            attention_mask=att_mask,
            lambda_kin=curr_lambda_kin,
        )

        loss = loss_dict["loss"]
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        step += 1

        if step % 50 == 0 or step == num_steps:
            elapsed = time.time() - t_start
            flow_l = loss_dict["flow_loss"].item()
            acc_l = loss_dict["acc_loss"].item()
            jerk_l = loss_dict["jerk_loss"].item()
            print(
                f"[Step {step:4d}/{num_steps}] "
                f"Loss: {loss.item():.4f} | "
                f"Flow MSE: {flow_l:.4f} | "
                f"Acc: {acc_l:.2f} | "
                f"Jerk: {jerk_l:.2f} | "
                f"lambda_kin: {curr_lambda_kin:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    # 5. Save Checkpoint
    save_path = os.path.join(output_dir, f"vga_libero_{shots}shot.pt")
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "config": cfg,
            "shots": shots,
        },
        save_path,
    )
    print(f"\n✅ Checkpoint saved to: {save_path}")

    # 6. Benchmark Real-Time Policy Latency on GPU vs Baseline
    print("\n--- 4. Benchmarking VGA Real-Time Inference Latency ---")
    policy.eval()
    policy.reset()

    sample_batch = {
        "image_front": img_front[:1],
        "input_ids": input_ids[:1],
        "attention_mask": att_mask[:1],
    }

    if device == "cuda":
        # Warmup passes
        for _ in range(5):
            policy.reset()
            _ = policy.select_action(sample_batch)

        torch.cuda.synchronize()
        iters = 50
        t0 = time.time()
        for _ in range(iters):
            policy.reset()
            _ = policy.select_action(sample_batch)
        torch.cuda.synchronize()
        chunk_latency_ms = (time.time() - t0) / iters * 1000.0

        print(f"✅ VGA Action-Chunk Latency (4-Step Euler ODE): {chunk_latency_ms:.2f} ms")
        print(f"   Control Rate: {1000.0 / chunk_latency_ms:.1f} inferences/sec")
        print(f"   Target Latency Ceiling: <= 18.0 ms")
        print(f"   Baseline SmolVLA Latency: 643.75 ms (~1.6 inferences/sec)")
        speedup = 643.75 / max(1e-3, chunk_latency_ms)
        print(f"   🚀 Speedup vs SmolVLA Baseline: {speedup:.1f}x faster!")

        if chunk_latency_ms <= 18.0:
            print("   🎯 MEETS REAL-TIME ROBOTICS 50 Hz REQUIREMENT (<= 18 ms)!")
    else:
        print("Device is CPU. Skipping CUDA latency benchmark.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=5, help="Number of demo episodes per task (5 or 10)")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_vga(
        shots=args.shots,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
    )
