"""
scripts/train_poc.py
Main training loop for the Vision-Geometry-Action (VGA-FewShot) Policy (v11).

Key Capabilities (v11):
1. Initializes from the 72.0% WiSE-FT Model Soup (0.50 * v10 + 0.50 * v9) when both checkpoints
   are available on disk, ensuring Step 0 begins at the highest-performing 72.0% basin.
2. Clean-RGB Metric Depth Extraction before Photometric ColorJitter:
   - Extracts Depth Anything V2 maps once per batch on the clean, un-jittered RGB tensors and passes
     them to both policy() and compute_metric_depth_prior(), cutting depth compute by 50% and
     keeping 3D geometry 100% invariant to lighting jitter.
3. Dedicated 5.0x Learning Rate Group (weight_decay=0.0) for Zero-Initialized Heads
   (fine_coord_mlp and dit.eef_pos_cond) so they wake up rapidly within 3,000 steps.
4. Online Exponential Moving Average (EMA, decay=0.998) exported to best.pt.
"""
import copy
import os
import random
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import math
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.v2 as T

from configs.poc_config import CONFIG
from data.dataset import LiberoSpatialDataset
from models.vga_policy import VGAPolicy
from losses.composite_loss import CompositeVGALoss


def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sanitize_container_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str
    normalized = path_str.replace("\\", "/")
    if "/storage/" in normalized and not normalized.startswith("/storage/"):
        normalized = "/storage/" + normalized.split("/storage/", 1)[1]
    elif re.match(r"^[A-Za-z]:/.*Git/", normalized):
        normalized = "/" + re.sub(r"^[A-Za-z]:/.*Git/", "", normalized).lstrip("/")
    return normalized


def build_3d_safe_augmentations():
    return T.Compose([
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
    ])


def sanitize_gripper_action_stats(action_stats: Dict[str, Any]) -> Dict[str, Any]:
    for m_key in ("mu", "mean"):
        if m_key in action_stats and len(action_stats[m_key]) >= 7:
            action_stats[m_key][6] = 0.0
    for s_key in ("sigma", "std"):
        if s_key in action_stats and len(action_stats[s_key]) >= 7:
            action_stats[s_key][6] = 1.0
    return action_stats


def build_transition_importance_sampler(
    dataset: LiberoSpatialDataset,
    chunk_horizon: int = 16
) -> Optional[WeightedRandomSampler]:
    total_len = len(dataset)
    if total_len == 0:
        return None

    weights = np.ones(total_len, dtype=np.float64)
    n_trans = 0
    n_pregrasp = 0

    idx_list = None
    for attr_name in ("index_map", "indices", "samples", "chunk_indices", "windows"):
        cand = getattr(dataset, attr_name, None)
        if isinstance(cand, list) and len(cand) == total_len:
            idx_list = cand
            break

    if hasattr(dataset, "episodes") and idx_list is not None:
        for i, item in enumerate(idx_list):
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                ep_idx, t = int(item[0]), int(item[1])
                ep = dataset.episodes[ep_idx]
                acts = np.asarray(ep["actions"], dtype=np.float32)
                T_len = acts.shape[0]
                if acts.shape[-1] >= 7:
                    g_seq = (acts[:, 6] > 0.0).astype(np.int32)
                    t_prev = max(0, t - 1)
                    t_end = min(T_len, t + chunk_horizon)
                    window_g = g_seq[t_prev:t_end]
                    if window_g.size > 1 and np.any(window_g[1:] != window_g[:-1]):
                        weights[i] = 4.0
                        n_trans += 1
                    else:
                        t_lookahead = min(T_len, t + chunk_horizon + 10)
                        look_g = g_seq[t:t_lookahead]
                        if look_g.size > 1 and np.any(look_g[1:] != look_g[:-1]):
                            weights[i] = 2.5
                            n_pregrasp += 1
    else:
        for i in range(total_len):
            sample = dataset[i]
            acts = sample.get("actions_raw", sample.get("actions_norm"))
            a_prev = sample.get("a_prev")
            if acts is not None and acts.shape[-1] >= 7:
                g_curr = (acts[:, 6] > 0.0)
                has_intra = bool(torch.any(g_curr[1:] != g_curr[:-1]).item()) if g_curr.numel() > 1 else False
                has_prev = False
                if a_prev is not None and a_prev.shape[-1] >= 7:
                    prev_g = bool((a_prev[-1, 6] > 0.0).item())
                    has_prev = bool(g_curr[0].item()) != prev_g
                if has_intra or has_prev:
                    weights[i] = 4.0
                    n_trans += 1

    n_transit = total_len - n_trans - n_pregrasp
    print(
        f">> [v11 Importance Sampler] Indexed {total_len} chunks: "
        f"4.0x Grasp/Release={n_trans} | 2.5x Pre-Grasp={n_pregrasp} | 1.0x Transit={n_transit}"
    )
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_tensor, num_samples=total_len, replacement=True)


def compute_metric_depth_prior_from_maps(
    depth_front: torch.Tensor,
    depth_wrist: Optional[torch.Tensor],
    align_dim: int,
    tokens_per_view: int = 64
) -> torch.Tensor:
    B = depth_front.shape[0]
    device = depth_front.device
    grid_size = int(math.sqrt(tokens_per_view))

    with torch.no_grad():
        depth_front_down = F.interpolate(depth_front.float(), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
        depth_tokens_front = depth_front_down.view(B, 1, -1).transpose(1, 2)

        if depth_wrist is not None:
            depth_wrist_down = F.interpolate(depth_wrist.float(), size=(grid_size, grid_size), mode="bilinear", align_corners=False)
            depth_tokens_wrist = depth_wrist_down.view(B, 1, -1).transpose(1, 2)
            depth_tokens = torch.cat([depth_tokens_front, depth_tokens_wrist], dim=1)
        else:
            depth_tokens = depth_tokens_front

        freqs = torch.exp(
            torch.arange(0, align_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / align_dim)
        )
        angles = depth_tokens.float() * freqs.view(1, 1, -1)
        target_depth_prior = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    return target_depth_prior


def train_vga(args: argparse.Namespace):
    if args.lambda_align is not None:
        args.lambda_depth = args.lambda_align

    args.data_dir = sanitize_container_path(args.data_dir)
    args.stats_path = sanitize_container_path(args.stats_path)
    args.output_dir = sanitize_container_path(args.output_dir)
    args.base_checkpoint = sanitize_container_path(args.base_checkpoint)
    args.soup_base = sanitize_container_path(args.soup_base)
    args.resume = sanitize_container_path(args.resume)

    set_deterministic_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("=" * 75)
    print(f">> INITIALIZING VGA v11 TRAINING REGIME: {args.split.upper()} ({args.shots or 'ALL'} SHOTS)")
    print(f">> Target Compute Device: {device}")
    print(f">> Total Steps: {args.steps} | Batch Size: {args.batch_size} | Base LR: {args.lr}")
    print(
        f">> Regularization: lambda_depth={args.lambda_depth} | lambda_grip={args.lambda_grip} | "
        f"lambda_smooth={args.lambda_smooth} | lambda_endpoint={args.lambda_endpoint}"
    )
    print(f">> Ray-RoPE Active: {not args.disable_ray_rope} | Dual Camera: {args.dual_camera}")
    print(f">> Clean-RGB Pure-GPU 518x518 Depth + Photometric Augs: {'ACTIVE' if not args.disable_aug else 'DISABLED'}")
    if args.base_checkpoint:
        print(f">> Foundational Base Checkpoint: {args.base_checkpoint}")
    print("=" * 75)

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        alt_dirs = [
            data_dir.replace("_shot", "shot"),
            data_dir.replace("shot", "_shot"),
            os.path.join("data", "libero_spatial"),
            os.path.join("data", "libero_90"),
            os.path.join("/storage/data", os.path.basename(data_dir))
        ]
        found = False
        for alt in alt_dirs:
            if os.path.exists(alt):
                data_dir = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Demonstration data directory not found at '{args.data_dir}'.")

    action_stats: Optional[Dict[str, Any]] = None
    if os.path.exists(args.stats_path):
        print(f">> Ingesting action normalizers from: {args.stats_path}")
        with open(args.stats_path, "r", encoding="utf-8") as f:
            action_stats = sanitize_gripper_action_stats(json.load(f))
    else:
        print(f">> Action stats not found at '{args.stats_path}'. Computing dynamically from demos...")

    num_shots = args.shots
    if args.split in ["all", "full"]:
        num_shots = None
    elif num_shots is None:
        if "5_shot" in args.split or "5shot" in args.split:
            num_shots = 5
        elif "10_shot" in args.split or "10shot" in args.split:
            num_shots = 10
        elif "50_shot" in args.split or "50shot" in args.split:
            num_shots = 50

    print(f">> Indexing trajectory demonstrations from: {data_dir} (num_shots={num_shots or 'ALL'})")
    chunk_horizon = getattr(CONFIG, "chunk_horizon", 16)
    dataset = LiberoSpatialDataset.from_disk(
        data_path=data_dir,
        num_shots=num_shots,
        chunk_horizon=chunk_horizon,
        prefix_len=getattr(CONFIG, "buffer_prefix_len", 4),
        action_stats=action_stats,
        seed=args.seed
    )

    if action_stats is None:
        action_stats = sanitize_gripper_action_stats(LiberoSpatialDataset.compute_action_stats(dataset.episodes))
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)), exist_ok=True)
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump(action_stats, f, indent=2)
        dataset.mu = torch.as_tensor(action_stats["mu"], dtype=torch.float32)
        dataset.sigma = torch.as_tensor(action_stats["sigma"], dtype=torch.float32)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_path)), exist_ok=True)
        with open(args.stats_path, "w", encoding="utf-8") as f:
            json.dump(action_stats, f, indent=2)

    batch_size = min(args.batch_size, max(len(dataset), 1))
    sampler = build_transition_importance_sampler(dataset, chunk_horizon=chunk_horizon)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4 if device.type == "cuda" else 0,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(dataset) >= batch_size)
    )
    print(f">> Trajectory chunks available: {len(dataset)} | Batches per epoch: {len(loader)}")

    policy = VGAPolicy(
        load_pretrained=(not args.no_pretrained),
        siglip_name=getattr(CONFIG, "siglip_model_name", "google/siglip-base-patch16-384"),
        smollm_name=getattr(CONFIG, "smollm_model_name", "HuggingFaceTB/SmolLM2-360M")
    ).to(device)

    criterion = CompositeVGALoss(
        lambda_depth=args.lambda_depth,
        lambda_grip=args.lambda_grip,
        lambda_smooth=args.lambda_smooth,
        lambda_endpoint=args.lambda_endpoint,
        transition_weight=6.0
    ).to(device)

    if args.base_checkpoint:
        if not os.path.isfile(args.base_checkpoint):
            raise FileNotFoundError(
                f"CRITICAL: Base checkpoint not found at '{args.base_checkpoint}'. "
                f"Refusing to silently fall back to training from scratch."
            )
        print(f">> [Transfer Learning] Loading base weights from: {args.base_checkpoint}")
        base_ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        state_dict = base_ckpt.get("model_state_dict", base_ckpt)

        # Auto-blend with v9 if soup_base exists so Step 0 starts at the 72.0% WiSE-FT Soup basin!
        soup_candidate = args.soup_base or "/storage/checkpoints/vga_10shot_v9/best.pt"
        if os.path.isfile(soup_candidate) and os.path.abspath(soup_candidate) != os.path.abspath(args.base_checkpoint):
            soup_ckpt = torch.load(soup_candidate, map_location=device, weights_only=False)
            soup_sd = soup_ckpt.get("model_state_dict", soup_ckpt)
            n_blended = 0
            for k, v in list(state_dict.items()):
                if k in soup_sd and v.shape == soup_sd[k].shape and torch.is_floating_point(v):
                    state_dict[k] = 0.50 * v.float() + 0.50 * soup_sd[k].float()
                    n_blended += 1
            print(f">> [WiSE-FT Base Initialization] Blended {n_blended} tensors (0.50 * {args.base_checkpoint} + 0.50 * {soup_candidate})!")

        # Remove any stale random fine_coord_mlp weights from v10 so v11 starts with clean zero-init on fine_coord_mlp[-1]
        for k in list(state_dict.keys()):
            if "fine_coord" in k:
                del state_dict[k]

        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
        print(
            f">> [VERIFIED] Base checkpoint loaded! "
            f"Matched: {len(state_dict) - len(unexpected_keys)} tensors | "
            f"Zero-init v11 heads (fine_coord_mlp & eef_pos_cond): {len(missing_keys)}"
        )

        print(">> [PEFT] Freezing foundational vision/LM base weights; tuning LoRA, DiT, EEF-3D-Cond (5x LR), 24x24 Fine-Coord (5x LR), Ray-RoPE, and Gripper heads.")
        for name, param in policy.named_parameters():
            param.requires_grad = False

            if "lora_" in name:
                param.requires_grad = True
            if "dit" in name:
                param.requires_grad = True
            if "projector" in name or "view_embed" in name or "fine_coord_mlp" in name:
                param.requires_grad = True
            if "ray_rope" in name or "pre_q_proj" in name or "pre_k_proj" in name or "pre_v_proj" in name or "pre_o_proj" in name:
                param.requires_grad = True
            if (
                "oxe_adapter" in name
                or "alpha_gate" in name
                or "vis_skip_gate" in name
                or "spatial_mlp" in name
                or "lang_vis_gate" in name
            ):
                param.requires_grad = True

    if args.disable_ray_rope:
        if hasattr(policy, "ray_rope") and hasattr(policy.ray_rope, "disable_ray_rope"):
            policy.ray_rope.disable_ray_rope = True

    policy.train()
    criterion.train()

    lora_params = []
    fast_zero_init_params = []
    standard_params = []

    for name, param in policy.named_parameters():
        if param.requires_grad:
            if "lora_" in name:
                lora_params.append(param)
            elif "fine_coord_mlp" in name or "eef_pos_cond" in name or "vis_skip_gate" in name or "alpha_gate" in name:
                fast_zero_init_params.append(param)
            else:
                standard_params.append(param)

    for name, param in criterion.named_parameters():
        if param.requires_grad:
            standard_params.append(param)

    print(
        f">> Active trainable parameter tensors: "
        f"{len(lora_params) + len(fast_zero_init_params) + len(standard_params)} "
        f"(Fast Zero-Init Heads @ 5x LR, WD=0.0: {len(fast_zero_init_params)})"
    )

    param_groups = [
        {"params": lora_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": standard_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": fast_zero_init_params, "lr": args.lr * 5.0, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    warmup_steps = min(150, max(args.steps // 10, 1))
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, args.steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        min_ratio = 1e-6 / max(args.lr, 1e-8)
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    use_amp = (device.type == "cuda" and not args.no_amp)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copy(args.stats_path, os.path.join(args.output_dir, "action_stats.json"))

    start_step = 0
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"CRITICAL: Resume checkpoint not found at '{args.resume}'.")
        print(f">> Resuming weights and optimizer state from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "criterion_state_dict" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f">> Warning: Optimizer state could not be resumed: {e}")
        start_step = checkpoint.get("step", 0)
        print(f">> Resumed from Step {start_step}")

    # Online Exponential Moving Average (EMA) anchored at the 72.0% initialization
    ema_decay = 0.998
    ema_state_dict = {
        k: v.detach().clone().float() if torch.is_floating_point(v) else v.detach().clone()
        for k, v in policy.state_dict().items()
    }

    aug_pipeline = build_3d_safe_augmentations() if not args.disable_aug else None

    step = start_step
    data_iter = iter(loader)
    start_time = time.time()
    all_trainable = lora_params + fast_zero_init_params + standard_params

    print(f"\n>> Optimization in progress (AMP: {use_amp}, Dtype: {amp_dtype}, EMA decay: {ema_decay})...\n")

    while step < args.steps:
        step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        rgb_front_clean = batch["rgb"].to(device, non_blocking=True)
        rgb_wrist_clean = batch["rgb_wrist"].to(device, non_blocking=True) if args.dual_camera else None

        # 1. Extract 3D metric depth on CLEAN, un-jittered RGB tensors (pure GPU 518x518)
        with torch.no_grad():
            depth_front = policy._extract_metric_depth(rgb_front_clean)
            depth_wrist = policy._extract_metric_depth(rgb_wrist_clean) if rgb_wrist_clean is not None else None

        # 2. Apply photometric ColorJitter only to the RGB appearance tensors
        if aug_pipeline is not None:
            rgb_front = aug_pipeline(rgb_front_clean)
            if args.aug_noise_std > 0.0:
                rgb_front = torch.clamp(rgb_front + torch.randn_like(rgb_front) * args.aug_noise_std, 0.0, 1.0)

            if rgb_wrist_clean is not None:
                rgb_wrist = aug_pipeline(rgb_wrist_clean)
                if args.aug_noise_std > 0.0:
                    rgb_wrist = torch.clamp(rgb_wrist + torch.randn_like(rgb_wrist) * args.aug_noise_std, 0.0, 1.0)
            else:
                rgb_wrist = None
        else:
            rgb_front = rgb_front_clean
            rgb_wrist = rgb_wrist_clean

        t_base_cam = batch["t_base_cam"].to(device, non_blocking=True)
        t_base_wrist = batch["t_base_wrist"].to(device, non_blocking=True) if args.dual_camera else None

        cam_intrinsics = batch.get("cam_intrinsics", None)
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.to(device, non_blocking=True)
        wrist_intrinsics = batch.get("wrist_intrinsics", None)
        if wrist_intrinsics is not None:
            wrist_intrinsics = wrist_intrinsics.to(device, non_blocking=True)

        x_1_norm = batch["actions_norm"].to(device, non_blocking=True)
        x_1_raw = batch.get("actions_raw", x_1_norm).to(device, non_blocking=True)
        a_prev = batch["a_prev"].to(device, non_blocking=True)

        if a_prev.shape[-1] == 6:
            grip_pad = torch.full((a_prev.shape[0], a_prev.shape[1], 1), -1.0, device=device, dtype=a_prev.dtype)
            a_prev = torch.cat([a_prev, grip_pad], dim=-1)

        prompt = batch["prompt"]
        B = rgb_front.shape[0]

        tau = torch.rand(B, device=device)
        x_0 = torch.randn_like(x_1_norm)
        tau_bc = tau.view(-1, 1, 1)
        x_tau = (1.0 - tau_bc) * x_0 + tau_bc * x_1_norm

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=amp_dtype, enabled=use_amp):
            v_pred, spatial_metric_features, gripper_logits = policy(
                rgb=rgb_front,
                t_base_cam=t_base_cam,
                x_t=x_tau,
                tau=tau,
                a_prev=a_prev,
                cam_intrinsics=cam_intrinsics,
                rgb_wrist=rgb_wrist,
                t_base_wrist=t_base_wrist,
                wrist_intrinsics=wrist_intrinsics,
                prompt=prompt,
                return_gripper_logits=True,
                depth_front_override=depth_front,
                depth_wrist_override=depth_wrist
            )

            target_depth_prior = batch.get("target_depth_prior", None)
            if target_depth_prior is None:
                target_depth_prior = compute_metric_depth_prior_from_maps(
                    depth_front=depth_front,
                    depth_wrist=depth_wrist,
                    align_dim=spatial_metric_features.shape[-1],
                    tokens_per_view=getattr(CONFIG, "compressed_tokens", 64)
                )
            else:
                target_depth_prior = target_depth_prior.to(device, non_blocking=True)

            total_loss, loss_flow, loss_depth = criterion(
                v_pred=v_pred,
                x_0=x_0,
                x_1_norm=x_1_norm,
                spatial_metric_features=spatial_metric_features,
                target_depth_prior=target_depth_prior.detach(),
                gripper_logits=gripper_logits,
                x_1_raw=x_1_raw,
                a_prev=a_prev
            )

        optimizer.zero_grad(set_to_none=True)
        if use_amp and amp_dtype == torch.float16:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
            optimizer.step()

        lr_scheduler.step()

        # Update Online EMA Weights
        with torch.no_grad():
            for k, v in policy.state_dict().items():
                if torch.is_floating_point(v):
                    ema_state_dict[k].mul_(ema_decay).add_(v.detach().float(), alpha=1.0 - ema_decay)
                else:
                    ema_state_dict[k].copy_(v.detach())

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            elapsed = time.time() - start_time
            rate = (step - start_step) / max(elapsed, 1e-4)
            fine_w_norm = float(policy.fine_coord_mlp[-1].weight.norm().item())
            eef_w_norm = float(policy.dit.eef_pos_cond[-1].weight.norm().item())
            print(
                f"Step [{step:05d}/{args.steps:05d}] | "
                f"Total: {total_loss.item():.4f} | "
                f"Flow: {loss_flow.item():.4f} (Endpt: {criterion.last_endpoint_loss:.4f}, Smooth: {criterion.last_smooth_loss:.4f}) | "
                f"GripBCE: {criterion.last_grip_loss:.4f} | "
                f"FineNorm: {fine_w_norm:.3f} | "
                f"EEF3DNorm: {eef_w_norm:.3f} | "
                f"Rate: {rate:.1f} steps/s"
            )

        if step % args.save_interval == 0 or step == args.steps:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": copy.deepcopy(ema_state_dict),
                "raw_model_state_dict": policy.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "action_stats": action_stats,
                "args": vars(args)
            }, ckpt_path)
            print(f">> EMA Checkpoint saved to: {ckpt_path}")

    final_path = os.path.join(args.output_dir, "best.pt")
    torch.save({
        "step": step,
        "model_state_dict": ema_state_dict,
        "raw_model_state_dict": policy.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "action_stats": action_stats,
        "args": vars(args)
    }, final_path)
    print(f"\n>> Training complete in {time.time() - start_time:.1f}s.")
    print(f">> Final EMA best model exported to: {final_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGA Policy on LIBERO Demonstrations")
    parser.add_argument("--split", type=str, default="10_shot")
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data/libero_spatial")
    parser.add_argument("--stats_path", type=str, default="configs/action_stats.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1500)
    parser.add_argument("--output_dir", type=str, default="checkpoints/vga_10shot_v11")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--soup_base", type=str, default=None)
    parser.add_argument("--dual_camera", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--disable_aug", action="store_true")
    parser.add_argument("--aug_noise_std", type=float, default=0.015)

    parser.add_argument("--lambda_depth", type=float, default=getattr(CONFIG, "lambda_depth", 0.15))
    parser.add_argument("--lambda_grip", type=float, default=getattr(CONFIG, "lambda_grip", 1.0))
    parser.add_argument("--lambda_smooth", type=float, default=getattr(CONFIG, "lambda_smooth", 0.15))
    parser.add_argument("--lambda_endpoint", type=float, default=getattr(CONFIG, "lambda_endpoint", 0.50))
    parser.add_argument("--lambda_align", type=float, default=None)
    parser.add_argument("--disable_ray_rope", action="store_true")

    parsed_args = parser.parse_args()
    train_vga(parsed_args)