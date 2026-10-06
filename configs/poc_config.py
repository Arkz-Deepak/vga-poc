"""
Configuration file for the VGA (Vision-Geometry-Action) PoC Model.

This file centralizes all architectural, training, and benchmark hyperparameters.
Every parameter has a specific physical and mathematical justification.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # 1. Vision & Space-to-Depth Projector
    vis_dim: int = 768            # SigLIP-B/16 output embedding dimension per patch token
    lm_dim: int = 576             # SmolLM2-135M hidden state dimension (language & fusion backbone)
    spatial_factor: int = 2       # Space-to-depth downscaling factor (16x16 patches -> 8x8 = 64 visual tokens)
    img_size: int = 256           # Input camera image resolution (256x256 RGB)
    patch_size: int = 16          # SigLIP patch size: 256 / 16 = 16 patches per axis (16x16 = 256 patches)
    num_visual_tokens: int = 64   # Final token count after space-to-depth compression (8x8 grid)
    num_lm_layers: int = 30       # Official SmolLM2-135M layer count (30 layers)
    pretrained_vision_model: str = "google/siglip-base-patch16-256"
    pretrained_lm_model: str = "HuggingFaceTB/SmolLM2-135M"
    pretrained: bool = True       # Load official pretrained weights from HuggingFace
    freeze_backbones: bool = True # Freeze backbones; only train Projector, RayRoPE & DiT expert

    # 2. Geometry & Ray-RoPE
    num_heads_rope: int = 8       # Multi-head attention heads in Ray-RoPE pre-pass
    origin_mlp_hidden: int = 256  # Hidden size of camera origin MLP(t_base)

    # 3. Action Expert (Diffusion Transformer / Flow Matching)
    action_horizon: int = 16      # Chunk horizon H: predicts 16 future steps
    execution_horizon: int = 8    # Receding horizon K: execute 8 steps before replanning (RTC / GROOVE)
    action_dim: int = 7           # 3 for delta pos (x, y, z), 3 for axis-angle rot (rx, ry, rz), 1 for gripper (g)
    prefix_len: int = 4           # Buffer tail length P: 4 waypoints (steps 13-16 of previous chunk)
    use_prefix_conditioning: bool = False  # Set to True when model was trained with prefix waypoints
    dit_layers: int = 12          # 12-layer Diffusion Transformer expert
    euler_steps: int = 4          # Number of function evaluations (NFE=4) for real-time 50 Hz control

    # 4. Kinematics & Loss Weights
    w_rot: float = 0.01           # Relative weighting of rotational jerk/acceleration vs translational
    beta_jerk: float = 0.5        # Relative weighting of jerk penalty compared to acceleration
    lambda_vq: float = 0.1        # Auxiliary VQ-depth classification loss weight
    kinematic_anneal_steps: int = 15000  # Warmup steps before full kinematic loss is applied
    max_lambda_kin: float = 0.05  # Maximum kinematic loss coefficient after warmup

    # 5. Controller & Inference
    dt: float = 0.02              # 20 ms per step -> 50 Hz control frequency
    schmitt_low: float = 0.40     # Gripper release threshold (g_bar < 0.40, raw < -0.20)
    schmitt_high: float = 0.60    # Gripper close threshold (g_bar > 0.60, raw > +0.20)
    min_hold_steps: int = 60      # Minimum steps (3.0s @ 20Hz) to keep gripper locked shut once grasped
    min_approach_steps: int = 25  # Pre-grasp guard: force gripper wide open during initial descent
    async_trigger_step: int = 11  # Step at which background inference begins (70% buffer consumption)

    # 6. Benchmark Tasks (LIBERO-Spatial)
    benchmark_tasks: List[str] = field(default_factory=lambda: ["all"])


# Default configuration instance
cfg = ModelConfig()
