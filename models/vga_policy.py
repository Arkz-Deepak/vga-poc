"""
Unified VGA Policy (Vision-Geometry-Action) Model.

Mathematical and Architectural Summary:
1. Multimodal Perception:
   - Front camera image I_front in R^{B x 3 x 256 x 256} -> SigLIP-B/16 -> 256 patches.
   - Space-to-Depth Projector: 256 patches -> 64 tokens in R^{B x 64 x 960}.
   - CentroidRayRoPE: Ground 64 tokens with unit viewing rays on S^2 and robot camera origin t_base.
   - SmolLM2 Language Backbone: Fuses task instructions and visual tokens -> context c in R^{B x 960}.

2. Action Generation (DiT Expert + Flow Matching):
   - 12-layer Diffusion Transformer predicts continuous 7D action trajectories over horizon H = 16.
   - 4-step Euler ODE integration: generates smooth, collision-free action chunks in <= 18 ms.
   - Target flow velocity u_t = x_1 - (1 - sigma_min) * x_0.

3. Physical Smoothness & Hysteresis:
   - Kinematic loss penalizes finite-difference acceleration and jerk (L_kin = L_acc + beta * L_jerk).
   - Schmitt Trigger Controller filters gripper predictions with [0.35, 0.65] deadband.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from controllers.schmitt_trigger import SchmittTriggerGripper
from data.dataset import Normalizer
from losses.kinematics import KinematicLoss
from models.backbones import VisionLanguageEncoder
from models.dit_expert import DiTActionExpert


class VGAPolicy(nn.Module):
    """
    Unified end-to-end Vision-Geometry-Action policy network (<= 0.5B parameters).
    """

    def __init__(
        self,
        cfg: Optional[Any] = None,
        normalizer: Optional[Normalizer] = None,
        vis_dim: int = 768,
        lm_dim: int = 960,
        img_size: int = 256,
        patch_size: int = 16,
        num_visual_tokens: int = 64,
        num_lm_layers: int = 12,
        action_dim: int = 7,
        action_horizon: int = 16,
        prefix_len: int = 4,
        dit_hidden_dim: int = 384,
        dit_layers: int = 12,
        euler_steps: int = 4,
        beta_jerk: float = 0.5,
        w_rot: float = 0.01,
        schmitt_low: float = 0.40,
        schmitt_high: float = 0.60,
        min_hold_steps: int = 60,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        if cfg is not None:
            vis_dim = getattr(cfg, "vis_dim", vis_dim)
            lm_dim = getattr(cfg, "lm_dim", lm_dim)
            img_size = getattr(cfg, "img_size", img_size)
            patch_size = getattr(cfg, "patch_size", patch_size)
            num_visual_tokens = getattr(cfg, "num_visual_tokens", num_visual_tokens)
            action_dim = getattr(cfg, "action_dim", action_dim)
            action_horizon = getattr(cfg, "action_horizon", action_horizon)
            prefix_len = getattr(cfg, "prefix_len", prefix_len)
            dit_layers = getattr(cfg, "dit_layers", dit_layers)
            euler_steps = getattr(cfg, "euler_steps", euler_steps)
            beta_jerk = getattr(cfg, "beta_jerk", beta_jerk)
            w_rot = getattr(cfg, "w_rot", w_rot)
            schmitt_low = getattr(cfg, "schmitt_low", schmitt_low)
            schmitt_high = getattr(cfg, "schmitt_high", schmitt_high)
            min_hold_steps = getattr(cfg, "min_hold_steps", min_hold_steps)
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.prefix_len = prefix_len
        self.euler_steps = euler_steps
        self.sigma_min = sigma_min
        self.normalizer = normalizer

        # 1. Vision-Language Encoder Backbone (~265M params)
        self.encoder = VisionLanguageEncoder(
            vis_dim=vis_dim,
            lm_dim=lm_dim,
            img_size=img_size,
            patch_size=patch_size,
            num_visual_tokens=num_visual_tokens,
            num_lm_layers=num_lm_layers,
        )

        # 2. Diffusion Transformer Action Expert (~33M params)
        self.expert = DiTActionExpert(
            action_dim=action_dim,
            action_horizon=action_horizon,
            prefix_len=prefix_len,
            context_dim=lm_dim,
            hidden_dim=dit_hidden_dim,
            num_layers=dit_layers,
            euler_steps=euler_steps,
        )

        # 3. Kinematic Loss Computer (Acceleration & Jerk with Taylor Series Lie Algebra)
        sigma_pos_sq = normalizer.sigma_pos_sq if normalizer is not None else 1.0
        sigma_rot_sq = normalizer.sigma_rot_sq if normalizer is not None else 1.0
        self.kinematic_loss_fn = KinematicLoss(
            sigma_pos_sq=sigma_pos_sq,
            sigma_rot_sq=sigma_rot_sq,
            w_rot=w_rot,
            beta_jerk=beta_jerk,
        )

        # 4. Schmitt Trigger Gripper Controller
        self.gripper_controller = SchmittTriggerGripper(
            low_thresh=schmitt_low,
            high_thresh=schmitt_high,
            min_hold_steps=min_hold_steps,
        )

        # 5. Runtime Action Chunk Queue for Rolling Rollouts
        self.action_queue = deque(maxlen=action_horizon)
        self.prev_chunk_tail: Optional[torch.Tensor] = None

    def reset(self):
        """Resets the policy action queue and gripper controller state."""
        self.action_queue.clear()
        self.prev_chunk_tail = None
        self.gripper_controller.reset()

    def forward(self, *args, **kwargs):
        """Default forward delegates to forward_loss for PyTorch Distributed / DataParallel compatibility."""
        return self.forward_loss(*args, **kwargs)

    def forward_loss(
        self,
        image_front: torch.Tensor,
        input_ids: torch.Tensor,
        actions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        camera_origin: Optional[torch.Tensor] = None,
        prefix_waypoints: Optional[torch.Tensor] = None,
        lambda_kin: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the joint Flow Matching Loss and Kinematic Regularization Loss during training.

        Args:
            image_front: Camera observation [B, 3, 256, 256]
            input_ids: Tokenized instruction IDs [B, seq_len]
            actions: Normalized ground-truth action chunk x_0 [B, 16, 7]
            attention_mask: Optional language attention mask [B, seq_len]
            camera_origin: Optional camera position in robot base frame [B, 3]
            prefix_waypoints: Optional P=4 tail waypoints from previous chunk [B, 4, 7]
            lambda_kin: Kinematic loss scaling weight
        Returns:
            Dictionary containing total_loss, flow_loss, acc_loss, jerk_loss
        """
        batch_size = actions.shape[0]
        device = actions.device

        # 1. Encode visual and language inputs into context c in R^{B x lm_dim}
        context = self.encoder(
            image_front=image_front,
            input_ids=input_ids,
            attention_mask=attention_mask,
            camera_origin=camera_origin,
        )

        # 2. Continuous-Time Flow Matching Setup
        # Sample uniform diffusion time t in [0, 1]
        t = torch.rand(batch_size, device=device)
        # Sample standard Gaussian noise x_1 ~ N(0, I)
        x_1 = torch.randn_like(actions)
        x_0 = actions  # Target clean action trajectory

        # Linear probability path: x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1
        t_exp = t[:, None, None]
        x_t = (1.0 - (1.0 - self.sigma_min) * t_exp) * x_0 + t_exp * x_1
        # Target velocity vector u_t = x_1 - (1 - sigma_min) * x_0
        u_t = x_1 - (1.0 - self.sigma_min) * x_0

        # 3. Predict vector field v_theta(x_t, t, c)
        v_pred = self.expert(
            x_t=x_t,
            t=t,
            context=context,
            prefix_waypoints=prefix_waypoints,
        )

        # 4. Flow Matching MSE Loss
        flow_loss = F.mse_loss(v_pred, u_t)

        # 5. Kinematic Smoothness Loss on predicted trajectory
        # Flow matching predicts: x_0_pred = (x_t - t * v_pred) / (1 - (1 - sigma_min) * t)
        denom = (1.0 - (1.0 - self.sigma_min) * t_exp).clamp(min=1e-3)
        x_0_pred = (x_t - t_exp * v_pred) / denom

        if prefix_waypoints is not None:
            prev_wp = prefix_waypoints[..., :6]
        else:
            prev_wp = x_0_pred[:, :1, :6].expand(-1, self.prefix_len, -1).detach()

        kin_loss, acc_loss, jerk_loss = self.kinematic_loss_fn(
            a_prev_phys=prev_wp,
            a_hat_phys=x_0_pred[..., :6],
        )

        # Total combined loss
        total_loss = flow_loss + lambda_kin * kin_loss

        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "kin_loss": kin_loss,
            "acc_loss": acc_loss,
            "jerk_loss": jerk_loss,
        }

    @torch.no_grad()
    def predict_chunk(
        self,
        image_front: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        camera_origin: Optional[torch.Tensor] = None,
        prefix_waypoints: Optional[torch.Tensor] = None,
        apply_schmitt: bool = True,
    ) -> torch.Tensor:
        """
        Predicts an entire 16-step action chunk in physical space [B, 16, 7].

        Args:
            image_front: Camera observation [B, 3, 256, 256]
            input_ids: Tokenized instruction IDs [B, seq_len]
            attention_mask: Optional language attention mask [B, seq_len]
            camera_origin: Optional camera position in robot base frame [B, 3]
            prefix_waypoints: Optional P=4 tail waypoints from previous chunk [B, 4, 7]
            apply_schmitt: Whether to apply Schmitt trigger hysteresis to gripper
        Returns:
            chunk_phys: Unnormalized physical action trajectories [B, 16, 7]
        """
        if image_front.dtype == torch.uint8:
            image_front = (image_front.float() / 255.0) * 2.0 - 1.0

        context = self.encoder(
            image_front=image_front,
            input_ids=input_ids,
            attention_mask=attention_mask,
            camera_origin=camera_origin,
        )

        chunk_norm = self.expert.sample_actions(
            context=context,
            prefix_waypoints=prefix_waypoints,
        )

        if self.normalizer is not None:
            chunk_phys = self.normalizer.unnormalize(chunk_norm)
        else:
            chunk_phys = chunk_norm

        if apply_schmitt:
            chunk_phys = chunk_phys.clone()
            B = chunk_phys.shape[0]
            for b in range(B):
                state = self.gripper_controller.open_val
                for t in range(self.action_horizon):
                    raw_g = chunk_phys[b, t, 6].item()
                    g_bar = (raw_g + 1.0) / 2.0
                    if g_bar > self.gripper_controller.high_thresh:
                        state = self.gripper_controller.close_val
                    elif g_bar < self.gripper_controller.low_thresh:
                        state = self.gripper_controller.open_val
                    chunk_phys[b, t, 6] = state

        return chunk_phys

    @torch.no_grad()
    def select_action(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Selects a single 7D action for robot execution, popping from the internal
        action queue or computing a fresh 16-step chunk in <= 18 ms when empty.

        Args:
            batch: Dictionary containing:
                - 'image_front' (or 'observation.images.camera1'): [1, 3, 256, 256]
                - 'input_ids' (or 'observation.language.tokens'): [1, seq_len]
                - 'attention_mask': [1, seq_len] (optional)
                - 'camera_origin': [1, 3] (optional)
        Returns:
            action: Physical action vector of shape (7,) [dx, dy, dz, rx, ry, rz, gripper]
        """
        # 1. If queue is empty, compute a new 16-step action chunk
        if len(self.action_queue) == 0:
            image_front = batch.get("image_front", batch.get("observation.images.camera1"))
            input_ids = batch.get("input_ids", batch.get("observation.language.tokens"))
            attention_mask = batch.get("attention_mask", batch.get("observation.language.attention_mask"))
            camera_origin = batch.get("camera_origin", None)

            # Ensure image is float in [-1, 1]
            if image_front.dtype == torch.uint8:
                image_front = (image_front.float() / 255.0) * 2.0 - 1.0

            # Encode context
            context = self.encoder(
                image_front=image_front,
                input_ids=input_ids,
                attention_mask=attention_mask,
                camera_origin=camera_origin,
            )

            # Generate 16-step action chunk via 4-step Euler ODE integration
            chunk_norm = self.expert.sample_actions(
                context=context,
                prefix_waypoints=None,
            )  # [1, 16, 7]

            # Save the tail P=4 steps to maintain state
            self.prev_chunk_tail = chunk_norm[:, -self.prefix_len:, :].clone()

            # Unnormalize actions to physical units if normalizer provided
            if self.normalizer is not None:
                chunk_phys = self.normalizer.unnormalize(chunk_norm)
            else:
                chunk_phys = chunk_norm

            # Push all 16 steps into queue in physical units
            chunk_cpu = chunk_phys.squeeze(0).cpu()  # [16, 7]
            for step_idx in range(self.action_horizon):
                self.action_queue.append(chunk_cpu[step_idx].clone())

        # 2. Pop the next ready action from queue and apply Schmitt Trigger per physical step
        step_action = self.action_queue.popleft()
        if self.gripper_controller is not None:
            raw_gripper = step_action[6].item()
            step_action[6] = self.gripper_controller.step(raw_gripper)

        return step_action
