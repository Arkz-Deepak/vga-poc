"""
configs/poc_config.py
Architectural hyperparameters, dimensional constants, and runtime configurations 
for the Vision-Geometry-Action (VGA-FewShot) Policy across 5-shot, 10-shot, and 
full demonstration regimes on LIBERO-Spatial.

Key Modifications for Metric Prior Plan:
1. Replaced legacy VQ-depth parameters with explicit L2 metric depth-consistency constraints.
2. Added `v_max_bounds` to physically clamp predicted velocity vectors in the receding horizon, 
   replacing the destructive in-network tanh formulation.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import torch


@dataclass(frozen=True)
class VGAConfig:
    # --- Image Resolutions & Patch Geometry ---
    sim_img_size: Tuple[int, int] = (256, 256)       # Raw camera frame from MuJoCo/LIBERO
    vis_input_size: Tuple[int, int] = (384, 384)     # Resized input for SigLIP (24x24 patches)
    patch_size: int = 16                             # ViT patch resolution
    raw_patches: int = 576                           # (384 // 16) ** 2 = 24 * 24
    spatial_factor: int = 3                          # Downsample factor for pixel_unshuffle (3x)
    compressed_tokens: int = 64                      # (24 // 3) ** 2 = 8 * 8 grid per camera view

    # --- 3D Geometric Grounding (Metric Ray-RoPE) ---
    use_ray_rope: bool = True                        # Enable Metric 3D Ray-RoPE
    rope_base: float = 10000.0                       # Base frequency for continuous 3D RoPE

    # --- Pretrained Perception & Language Backbones ---
    siglip_model_name: str = "google/siglip-base-patch16-384"
    smollm_model_name: str = "HuggingFaceTB/SmolLM2-360M"
    vis_dim: int = 768                               # SigLIP-B/16 embedding dimension
    lm_dim: int = 960                                # SmolLM2-360M hidden state dimension
    unshuffled_channels: int = 6912                  # vis_dim * (spatial_factor ** 2) = 768 * 9
    rope_heads: int = 8                              # CentroidRayRoPE pre-pass attention heads

    # --- Parameter-Efficient LoRA Fine-Tuning ---
    lora_r: int = 16                                 # Rank for parameter-efficient adaptation
    lora_alpha: int = 32                             # Scaling factor for LoRA updates

    # --- Action Trajectory & Flow Matching DiT Expert ---
    action_dim: int = 7                              # [dx, dy, dz, rx, ry, rz, gripper]
    chunk_horizon: int = 16                          # Prediction chunk horizon H
    exec_horizon: int = 8                            # Receding horizon execution steps before replanning
    buffer_prefix_len: int = 4                       # History waypoints P (steps t-4 .. t-1)
    dit_hidden_dim: int = 512                        # Action expert hidden dimension
    dit_layers: int = 12                             # Action expert transformer depth
    dit_heads: int = 8                               # Action expert multi-head attention count

    # --- Kinematic Limits (Post-Network Rejection) ---
    # Limits maximum instantaneous delta [dx, dy, dz, rx, ry, rz, gripper]
    v_max_bounds: Tuple[float, float, float, float, float, float, float] = (
        0.05, 0.05, 0.05,   # Max 5cm translation per chunk step
        0.25, 0.25, 0.25,   # Max 0.25 rad rotation per chunk step
        1.0                 # Gripper state bounded to [-1.0, 1.0]
    )

    # --- Loss Weights & Regularization ---
    w_rot: float = 0.01                              # Relative rotation loss weight
    beta_jerk: float = 0.5                           # Ratio of jerk to acceleration regularizer
    lambda_kin_max: float = 0.01                     # Trajectory acceleration and jerk loss scalar
    kinematic_warmup_steps: int = 2000               # Steps to linearly ramp kinematic loss
    lambda_join: float = 0.02                        # Chunk-join boundary continuity loss scalar
    
    # Replaced shadow-matching proxy with true Metric Depth-Consistency MSE
    lambda_depth: float = 0.15                       # Weight for L2 depth map matching
    
    # Deprecated Variables (Retained to prevent import breaks in older scripts)
    num_depth_bins: int = 32                         
    lambda_vq: float = 0.0                           
    vq_codebook_size: int = 256                      

    # --- Synthetic Drift Injection ---
    max_trans_drift: float = 0.015                   # 15 mm maximum spatial perturbation
    max_rot_drift_deg: float = 3.0                   # 3 degrees maximum orientation perturbation
    drift_ramp_steps: int = 3                        # Perturbation decay horizon (steps)

    # --- Controller Thresholds & Observation Flags ---
    dual_camera: bool = True                         # Enable Front ('agentview') + Wrist ('eye-in-hand') streams
    schmitt_low: float = -0.40                       # Gripper open hysteresis threshold
    schmitt_high: float = 0.40                       # Gripper close hysteresis threshold
    async_trigger_step: int = 11                     # Replanning trigger step for background thread
    max_eval_steps: int = 280                        # Maximum rollout duration per episode in simulation
    use_depth_input: bool = False                    # Test-time RGB only (offline metric prior is synthesized inside VLA)
    drop_depth_head_at_eval: bool = True             # Drop auxiliary heads during inference

    # --- Front Camera Intrinsics (Static Third-Person View) ---
    default_intrinsics: Dict[str, float] = field(default_factory=lambda: {
        "fx": 200.0,
        "fy": 200.0,
        "cx": 128.0,
        "cy": 128.0
    })

    # --- Wrist Camera Intrinsics (Dynamic Eye-in-Hand View) ---
    wrist_intrinsics: Dict[str, float] = field(default_factory=lambda: {
        "fx": 150.0,
        "fy": 150.0,
        "cx": 128.0,
        "cy": 128.0
    })

CONFIG = VGAConfig()
ModelConfig = VGAConfig
cfg = CONFIG
