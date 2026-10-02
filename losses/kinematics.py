"""
Lie Algebra Kinematics & Motion Smoothing Losses.

Role in VGA (Vision-Geometry-Action) Architecture:
-------------------------------------------------
Traditional diffusion and regression policies predict actions step-by-step or in chunks
without enforcing laws of physical dynamics. When executed on a physical or simulated
robot arm, this results in:
1. High-frequency joint vibration ("motor chatter").
2. Abrupt velocity jumps and acceleration spikes between consecutive steps and chunk boundaries.
3. Dropped objects and wear-and-tear on actuators.

Solution:
---------
1. Safe Lie Algebra Conversion (SO(3) -> so(3)):
   - Standard axis-angle conversion divides by sin(theta).
   - Near theta = 0 (small wrist motions), sin(theta) -> 0 causing 0/0 NaN explosions.
   - We implement a 4th-order Taylor expansion guard:
     theta / sin(theta) = 1 + (theta^2)/6 + 7*(theta^4)/360 + O(theta^6)
     guaranteeing strictly finite, stable gradients.

2. Physical Kinematic Loss (Acceleration & Jerk Regularization):
   - Computes 1st, 2nd, and 3rd order finite differences across the trajectory.
   - Penalizes acceleration (L_acc) and jerk (L_jerk, rate of change of acceleration).
   - Smooths trajectory handoffs between the previous executed buffer and new predictions.
"""

from typing import Tuple
import torch
import torch.nn as nn


def safe_rotation_matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Converts 3x3 rotation matrices in SO(3) to 3D axis-angle vectors in Lie algebra so(3)
    using a 4th-order Taylor series approximation around theta=0 to eliminate NaN singularities.

    Args:
        R: Rotation matrix tensor of shape [..., 3, 3]
        eps: Small epsilon clamp to prevent numerical boundary violations in acos

    Returns:
        r: Axis-angle rotation vector of shape [..., 3], where norm(r) = theta (angle in radians)
           and r / norm(r) = rotation unit axis u.
    """
    # Trace of 3x3 matrix: tr(R) = R_00 + R_11 + R_22 = 1 + 2 * cos(theta)
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    
    # cos(theta) = (tr(R) - 1) / 2, clamped strictly to [-1 + eps, 1 - eps]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    # 4th-order Taylor series for f(theta) = theta / sin(theta) near theta = 0:
    # theta / sin(theta) = 1 + (theta^2)/6 + 7*(theta^4)/360
    small_angle_factor = 1.0 + (theta ** 2) / 6.0 + (7.0 * (theta ** 4)) / 360.0
    normal_factor = theta / (sin_theta + 1e-8)

    # Use Taylor guard when absolute angle is below 1e-4 radians (approx 0.0057 degrees)
    factor = torch.where(theta.abs() < 1e-4, small_angle_factor, normal_factor)

    # Off-diagonal skews: (R - R^T) corresponds to 2 * sin(theta) * [u]_x
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]

    # r = 0.5 * factor * [rx, ry, rz]
    r = 0.5 * factor.unsqueeze(-1) * torch.stack([rx, ry, rz], dim=-1)
    return r


def compute_kinematics(actions: torch.Tensor, dt: float = 0.02) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes finite-difference acceleration and jerk across an action sequence.

    Args:
        actions: Stitched trajectory tensor of shape [B, T, D] (e.g. D=6: 3 pos + 3 rot)
        dt: Time delta between control steps (0.02s for 50 Hz)

    Returns:
        acc: Acceleration tensor of shape [B, T - 1, D]
        jerk: Jerk tensor of shape [B, T - 2, D]
    """
    # Step velocity / delta between consecutive waypoints
    # acc = delta_action / dt (or discrete difference between steps)
    diff1 = actions[:, 1:] - actions[:, :-1]
    acc = diff1 / dt

    # Jerk is rate of change of acceleration: d(acc) / dt
    diff2 = acc[:, 1:] - acc[:, :-1]
    jerk = diff2 / dt

    return acc, jerk


class KinematicLoss(nn.Module):
    """
    Physics-informed kinematic loss penalizing acceleration and jerk across stitched action chunks.
    """
    def __init__(
        self,
        sigma_pos_sq: float = 1.0,
        sigma_rot_sq: float = 1.0,
        w_rot: float = 0.01,
        beta_jerk: float = 0.5
    ):
        super().__init__()
        self.sigma_pos_sq = sigma_pos_sq
        self.sigma_rot_sq = sigma_rot_sq
        self.w_rot = w_rot
        self.beta_jerk = beta_jerk

    def forward(
        self,
        a_prev_phys: torch.Tensor,
        a_hat_phys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            a_prev_phys: Last P=4 executed physical waypoints from buffer tail [B, P, 6]
            a_hat_phys: Model predicted physical waypoints for next H=16 steps [B, H, 6]
                        where 6 dimensions are [delta_x, delta_y, delta_z, r_x, r_y, r_z]

        Returns:
            l_kin: Combined weighted kinematic loss (L_acc + beta * L_jerk)
            l_acc: Pure acceleration loss component
            l_jerk: Pure jerk loss component
        """
        # Stitch active buffer tail with predicted future actions:
        # [B, 4, 6] + [B, 16, 6] -> [B, 20, 6]
        a_stitched = torch.cat([a_prev_phys, a_hat_phys], dim=1)

        # Decompose into 3D translation (pos) and 3D rotation (rot)
        delta_p = a_stitched[..., :3]
        r = a_stitched[..., 3:6]

        # 1st order differences (velocity/acceleration in discrete formulation)
        acc_p = delta_p[:, 1:] - delta_p[:, :-1]
        acc_r = r[:, 1:] - r[:, :-1]

        l_acc = torch.mean(
            (torch.norm(acc_p, dim=-1) ** 2) / (self.sigma_pos_sq + 1e-8) +
            self.w_rot * (torch.norm(acc_r, dim=-1) ** 2) / (self.sigma_rot_sq + 1e-8)
        )

        # 2nd order differences (jerk)
        jerk_p = acc_p[:, 1:] - acc_p[:, :-1]
        jerk_r = acc_r[:, 1:] - acc_r[:, :-1]

        l_jerk = torch.mean(
            (torch.norm(jerk_p, dim=-1) ** 2) / (self.sigma_pos_sq + 1e-8) +
            self.w_rot * (torch.norm(jerk_r, dim=-1) ** 2) / (self.sigma_rot_sq + 1e-8)
        )

        l_kin = l_acc + self.beta_jerk * l_jerk
        return l_kin, l_acc, l_jerk
