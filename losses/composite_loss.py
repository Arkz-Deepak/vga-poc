"""
losses/composite_loss.py
Multi-objective composite loss (v10):
  1. Receding-Horizon Exponentially Weighted 6-DoF Continuous Flow Matching (CFM) Velocity Loss
  2. Cumulative Trajectory Path & 16-Step Chunk Endpoint Loss (L_traj + L_endpoint on v_pred - target_v)
  3. Kinematic Velocity-Smoothness / Intra-Chunk Acceleration Loss (L_smooth)
  4. Transition-Weighted Decoupled Binary Cross-Entropy (BCE) Gripper Classification Loss
  5. Metric Depth-Consistency (L2/MSE) Spatial Forcing Loss

Key Updates (v10):
- Supervises the cumulative velocity path cumsum(v_pred - target_v, dim=1) / H and the 16-step mean
  chunk displacement (v_pred[:, :, :3] - target_v[:, :, :3]).mean(dim=1) directly in velocity space
  (avoiding (1 - tau)^2 gradient vanishing at tau -> 1), eliminating the 2.5-3.5cm endpoint undershoot
  at the bowl rim and plate center.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeVGALoss(nn.Module):
    def __init__(
        self,
        lambda_depth: float = 1.0,
        lambda_grip: float = 1.0,
        lambda_smooth: float = 0.15,
        lambda_endpoint: float = 0.50,
        transition_weight: float = 6.0,
        exec_horizon: int = 4,
    ):
        super().__init__()
        self.lambda_depth = lambda_depth
        self.lambda_grip = lambda_grip
        self.lambda_smooth = lambda_smooth
        self.lambda_endpoint = lambda_endpoint
        self.transition_weight = transition_weight
        self.exec_horizon = exec_horizon
        self.last_grip_loss: float = 0.0
        self.last_smooth_loss: float = 0.0
        self.last_endpoint_loss: float = 0.0

    def _build_horizon_weights(self, horizon: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        steps = torch.arange(horizon, device=device, dtype=torch.float32)
        raw_weights = torch.where(
            steps < self.exec_horizon,
            torch.full_like(steps, 1.60),
            1.60 * (0.90 ** (steps - float(self.exec_horizon - 1)))
        )
        norm_weights = raw_weights / raw_weights.mean().clamp(min=1e-6)
        return norm_weights.to(dtype=dtype).unsqueeze(0)

    def forward(
        self,
        v_pred: torch.Tensor,
        x_0: torch.Tensor,
        x_1_norm: torch.Tensor,
        spatial_metric_features: torch.Tensor,
        target_depth_prior: torch.Tensor,
        gripper_logits: Optional[torch.Tensor] = None,
        x_1_raw: Optional[torch.Tensor] = None,
        a_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, _ = v_pred.shape
        target_v = x_1_norm - x_0
        horizon_weights = self._build_horizon_weights(H, v_pred.device, v_pred.dtype)

        # -------------------------------------------------------------
        # 1. Receding-Horizon Flow Matching Loss
        # -------------------------------------------------------------
        if gripper_logits is not None and v_pred.shape[-1] >= 7:
            arm_sq_err = torch.sum((v_pred[:, :, :6] - target_v[:, :, :6]) ** 2, dim=-1)
            grip_flow_err = 0.1 * ((v_pred[:, :, 6] - target_v[:, :, 6]) ** 2)
            l_flow_base = torch.mean((arm_sq_err + grip_flow_err) * horizon_weights)
        else:
            flow_sq_err = torch.sum((v_pred - target_v) ** 2, dim=-1)
            l_flow_base = torch.mean(flow_sq_err * horizon_weights)

        # -------------------------------------------------------------
        # 2. v10 Cumulative Path & 16-Step Chunk Endpoint Loss (Un-damped across tau)
        # -------------------------------------------------------------
        if self.lambda_endpoint > 0.0:
            v_err_spatial = v_pred[:, :, :6] - target_v[:, :, :6]  # [B, H, 6]
            cum_err = torch.cumsum(v_err_spatial, dim=1) / float(max(H, 1))
            l_traj = torch.mean(torch.sum(cum_err ** 2, dim=-1))

            end_err_xyz = torch.mean(v_err_spatial[:, :, :3], dim=1)  # [B, 3] net 16-step XYZ drift
            l_end = torch.mean(torch.sum(end_err_xyz ** 2, dim=-1))

            l_endpoint = 0.5 * l_traj + l_end
            self.last_endpoint_loss = float(l_endpoint.detach().item())
        else:
            l_endpoint = torch.zeros_like(l_flow_base)
            self.last_endpoint_loss = 0.0

        # -------------------------------------------------------------
        # 3. Kinematic Velocity-Smoothness Loss
        # -------------------------------------------------------------
        if H > 1 and self.lambda_smooth > 0.0:
            dv_pred = v_pred[:, 1:, :6] - v_pred[:, :-1, :6]
            dv_target = target_v[:, 1:, :6] - target_v[:, :-1, :6]
            l_smooth = F.mse_loss(dv_pred, dv_target)
            self.last_smooth_loss = float(l_smooth.detach().item())
        else:
            l_smooth = torch.zeros_like(l_flow_base)
            self.last_smooth_loss = 0.0

        l_flow = l_flow_base + (self.lambda_endpoint * l_endpoint) + (self.lambda_smooth * l_smooth)

        # -------------------------------------------------------------
        # 4. Transition-Weighted Binary Cross-Entropy (BCE) Gripper Loss
        # -------------------------------------------------------------
        if gripper_logits is not None:
            grip_source = x_1_raw[:, :, 6] if (x_1_raw is not None and x_1_raw.shape[-1] >= 7) else x_1_norm[:, :, 6]
            target_grip_binary = (grip_source > 0.0).to(dtype=torch.float32)

            intra_flips = torch.zeros_like(target_grip_binary, dtype=torch.bool)
            if H > 1:
                step_changes = target_grip_binary[:, 1:] != target_grip_binary[:, :-1]
                intra_flips[:, 1:] = torch.cummax(step_changes.to(torch.int8), dim=1)[0].to(torch.bool)

            if a_prev is not None and a_prev.ndim == 3 and a_prev.shape[-1] >= 7:
                prev_grip_binary = (
                    a_prev[:, -1:, 6].to(device=target_grip_binary.device) > 0.0
                ).to(dtype=torch.float32)
                flips_from_prev = (target_grip_binary != prev_grip_binary)
                is_transition = flips_from_prev | intra_flips
            else:
                is_transition = intra_flips

            bce_weights = torch.where(
                is_transition,
                torch.full_like(target_grip_binary, self.transition_weight),
                torch.ones_like(target_grip_binary)
            )

            l_grip = F.binary_cross_entropy_with_logits(
                gripper_logits.to(dtype=torch.float32),
                target_grip_binary,
                weight=bce_weights,
                reduction="mean"
            ).to(dtype=l_flow.dtype)
            self.last_grip_loss = float(l_grip.detach().item())
        else:
            l_grip = torch.zeros_like(l_flow)
            self.last_grip_loss = 0.0

        # -------------------------------------------------------------
        # 5. Metric Depth-Consistency Loss & Composite Objective
        # -------------------------------------------------------------
        target_depth_prior = target_depth_prior.detach()
        l_depth = F.mse_loss(spatial_metric_features, target_depth_prior)

        l_unified = l_flow + (self.lambda_grip * l_grip) + (self.lambda_depth * l_depth)

        return l_unified, l_flow, l_depth