"""
Phase 1 Verification Script: Synthetic Tensor Sanity Pass.

This script tests all Phase 1 architectural modules locally using synthetic tensors:
1. UnifiedSpaceToDepthProjector: 576 tokens -> 64 tokens without shape mismatch.
2. CentroidRayRoPE: Ray geometry generation, camera translation embedding, and attention pre-pass.
3. Lie Algebra Kinematics: Safe axis-angle extraction near theta=0 (Taylor guard test) and kinematic loss.
4. Autograd Check: Ensures loss backpropagation produces finite, non-NaN gradients.
"""

import sys
import os
import math

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from configs.poc_config import cfg
from models.projector import UnifiedSpaceToDepthProjector
from models.ray_rope import CentroidRayRoPE
from losses.kinematics import safe_rotation_matrix_to_axis_angle, KinematicLoss


def test_projector():
    print("\n--- [1/4] Testing UnifiedSpaceToDepthProjector ---")
    batch_size = 2
    vis_dim = cfg.vis_dim      # 768
    lm_dim = cfg.lm_dim        # 576
    spatial_factor = cfg.spatial_factor  # 2
    grid_dim = cfg.img_size // cfg.patch_size  # 256 / 16 = 16
    num_patches = grid_dim * grid_dim          # 16x16 = 256 patches from SigLIP-256

    projector = UnifiedSpaceToDepthProjector(
        vis_dim=vis_dim,
        lm_dim=lm_dim,
        spatial_factor=spatial_factor
    )

    # Synthetic vision encoder output: [B, 256, 768]
    dummy_vis_patches = torch.randn(batch_size, num_patches, vis_dim)
    print(f"Input visual patches shape: {dummy_vis_patches.shape}")

    out_tokens = projector(dummy_vis_patches)
    print(f"Output compressed tokens shape: {out_tokens.shape}")

    assert out_tokens.shape == (batch_size, cfg.num_visual_tokens, lm_dim), (
        f"Expected shape ({batch_size}, {cfg.num_visual_tokens}, {lm_dim}), got {out_tokens.shape}"
    )
    assert not torch.isnan(out_tokens).any(), "Projector output contains NaN!"
    assert not torch.isinf(out_tokens).any(), "Projector output contains Inf!"
    print("Projector test PASSED! (256 patches -> 64 tokens with 0 NaN)")


def test_ray_rope():
    print("\n--- [2/4] Testing CentroidRayRoPE & Geometry Generation ---")
    batch_size = 2
    lm_dim = cfg.lm_dim  # 960

    ray_rope = CentroidRayRoPE(lm_dim=lm_dim, num_heads=cfg.num_heads_rope)

    # 1. Camera intrinsics (standard pinhole camera 256x256)
    fx = fy = 200.0
    cx = cy = 128.0
    K = torch.tensor([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # 2. Camera extrinsics (identity rotation for test)
    R_base_cam = torch.eye(3, dtype=torch.float32)

    # Compute 3D viewing rays for the 8x8 centroid grid
    rays = CentroidRayRoPE.compute_centroid_rays(K, R_base_cam, img_w=256, img_h=256)
    print(f"Computed centroid rays shape: {rays.shape}")
    assert rays.shape == (64, 3), f"Expected rays shape (64, 3), got {rays.shape}"

    # Verify all rays are unit vectors on S^2 (norm == 1.0)
    ray_norms = torch.norm(rays, dim=-1)
    assert torch.allclose(ray_norms, torch.ones_like(ray_norms), atol=1e-5), "Rays are not unit length!"
    print("Ray geometry test PASSED! (All 64 centroid rays have unit norm on S^2)")

    # 3. Test forward pass with camera translation t_base_cam
    dummy_tokens = torch.randn(batch_size, 64, lm_dim)
    t_base_cam = torch.tensor([[0.25, -0.10, 0.50], [0.30, -0.05, 0.45]], dtype=torch.float32)

    rays_batched = rays.unsqueeze(0).expand(batch_size, -1, -1)
    out_grounded = ray_rope(dummy_tokens, t_base_cam, rays_base=rays_batched)
    print(f"Grounded visual tokens shape: {out_grounded.shape}")

    assert out_grounded.shape == (batch_size, 64, lm_dim)
    assert not torch.isnan(out_grounded).any(), "Ray-RoPE output contains NaN!"
    assert not torch.isinf(out_grounded).any(), "Ray-RoPE output contains Inf!"
    print("CentroidRayRoPE forward test PASSED!")


def test_lie_algebra_and_kinematics():
    print("\n--- [3/4] Testing Safe Lie Algebra Axis-Angle (Taylor Guard) ---")

    # Case A: Standard rotation (90 degrees around Z axis)
    R_90z = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=torch.float32)
    r_90z = safe_rotation_matrix_to_axis_angle(R_90z)
    print(f"90-deg Z axis-angle output: {r_90z.tolist()}")
    expected_angle = math.pi / 2.0
    actual_angle = torch.norm(r_90z).item()
    assert abs(actual_angle - expected_angle) < 1e-4, f"Expected {expected_angle}, got {actual_angle}"

    # Case B: Zero rotation (Identity matrix - the notorious singularity where standard code divides by sin(0)=0)
    R_identity = torch.eye(3, dtype=torch.float32)
    r_zero = safe_rotation_matrix_to_axis_angle(R_identity)
    print(f"Zero-rotation (Identity) output: {r_zero.tolist()}")
    assert not torch.isnan(r_zero).any(), "Identity matrix produced NaN in axis-angle conversion!"
    assert torch.allclose(r_zero, torch.zeros(3), atol=1e-6), "Identity matrix did not produce zero rotation!"

    # Case C: Microscopic rotation (theta = 1e-5 radians)
    theta_micro = 1e-5
    cos_t = math.cos(theta_micro)
    sin_t = math.sin(theta_micro)
    R_micro = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ], dtype=torch.float32)
    r_micro = safe_rotation_matrix_to_axis_angle(R_micro)
    print(f"Micro-rotation (1e-5 rad) output norm: {torch.norm(r_micro).item():.8f}")
    assert not torch.isnan(r_micro).any(), "Micro-rotation produced NaN!"
    assert abs(torch.norm(r_micro).item() - theta_micro) < 1e-7, "Taylor approximation error too large!"
    print("Lie algebra Taylor guard test PASSED! (Zero NaN across all singularities)")


def test_kinematic_loss_and_gradients():
    print("\n--- [4/4] Testing KinematicLoss & Autograd Backward Pass ---")
    batch_size = 2
    prefix_len = cfg.prefix_len        # P = 4 waypoints from buffer tail
    action_horizon = cfg.action_horizon  # H = 16 waypoints predicted
    action_dim = 6                     # 3 pos + 3 rot

    kin_loss_fn = KinematicLoss(
        sigma_pos_sq=1.0,
        sigma_rot_sq=1.0,
        w_rot=cfg.w_rot,
        beta_jerk=cfg.beta_jerk
    )

    # Previous buffer tail (fixed, no grad)
    a_prev_phys = torch.randn(batch_size, prefix_len, action_dim)

    # Model predicted actions (requires grad)
    a_hat_phys = torch.randn(batch_size, action_horizon, action_dim, requires_grad=True)

    l_kin, l_acc, l_jerk = kin_loss_fn(a_prev_phys, a_hat_phys)
    print(f"Kinematic loss values: Total={l_kin.item():.4f}, Acc={l_acc.item():.4f}, Jerk={l_jerk.item():.4f}")

    assert not torch.isnan(l_kin), "Kinematic loss is NaN!"
    assert not torch.isinf(l_kin), "Kinematic loss is Inf!"
    assert l_kin > 0, "Kinematic loss must be positive!"

    # Test backward pass to verify non-zero finite gradients
    l_kin.backward()
    assert a_hat_phys.grad is not None, "Gradients were not computed!"
    assert not torch.isnan(a_hat_phys.grad).any(), "Gradient contains NaN!"
    assert not torch.isinf(a_hat_phys.grad).any(), "Gradient contains Inf!"
    assert (a_hat_phys.grad != 0).any(), "Gradient is all zeros!"
    print(f"Autograd backward pass PASSED! (Grad norm: {torch.norm(a_hat_phys.grad).item():.4f})")


def run_all_tests():
    print("================================================================")
    print("      VGA PoC Phase 1: Local Staging & Architecture Check       ")
    print("================================================================")
    test_projector()
    test_ray_rope()
    test_lie_algebra_and_kinematics()
    test_kinematic_loss_and_gradients()
    print("\n================================================================")
    print("  ALL PHASE 1 SANITY CHECKS PASSED SUCCESSFULLY WITH ZERO NAH!  ")
    print("================================================================\n")


if __name__ == "__main__":
    run_all_tests()
