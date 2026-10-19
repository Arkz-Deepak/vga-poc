"""
models/ray_rope.py

Continuous 3D Rotary Position Embeddings (Ray-RoPE) grounded in Metric Depth and Full SE(3) Pose.

Modifications for Metric Grounding & Cross-Node Determinism:
1. `generate_patch_rays` computes camera origins and SE(3)-rotated ray directions in strict IEEE-754
   float32 precision (disabling bfloat16 autocast during pinhole inversion and R @ d_cam rotation).
2. `ContinuousRayRoPE` resizes the deterministic depth map to match the patch grid resolution and
   reconstructs metric 3D coordinates (Origin + Direction * Depth) in float32 before computing
   trigonometric RoPE frequencies.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_patch_rays(
    t_base_cam: torch.Tensor,
    cam_intrinsics: torch.Tensor,
    img_size: int = 384,
    patch_size: int = 16
) -> torch.Tensor:
    """
    Generates 3D ray origins and unnormalized directions for each image patch in strict float32.
    Returns tensor of shape [B, N, 6] where [..., :3] is Origin and [..., 3:] is Direction.
    """
    B = t_base_cam.shape[0]
    device = t_base_cam.device

    with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=False):
        t_base_f32 = t_base_cam.float()
        intrinsics_f32 = cam_intrinsics.float()

        # Calculate grid size (e.g., 128 // 16 = 8 -> 64 patches)
        grid_size = img_size // patch_size
        num_patches = grid_size * grid_size

        # Create pixel grid representing the centers of patches in float32
        u = torch.arange(patch_size // 2, img_size, patch_size, device=device, dtype=torch.float32)
        v = torch.arange(patch_size // 2, img_size, patch_size, device=device, dtype=torch.float32)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")

        # Pixels in homogeneous coordinates: [B, 3, N]
        uv = torch.stack([grid_u.flatten(), grid_v.flatten(), torch.ones_like(grid_u.flatten())], dim=-1)
        uv = uv.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)

        # Inverse intrinsics to get camera-frame unnormalized rays (where Z=1)
        inv_K = torch.linalg.inv(intrinsics_f32)
        rays_cam_unnorm = torch.bmm(inv_K, uv)  # [B, 3, N]

        # Transform to base frame using full SE(3) camera extrinsics (Rotation R + Translation T)
        if t_base_f32.shape[-1] == 4:
            R = t_base_f32[:, :3, :3]
            T = t_base_f32[:, :3, 3].unsqueeze(-1).expand(-1, -1, num_patches)  # [B, 3, N]
        else:
            R = t_base_f32[:, :3, :3]
            T = torch.zeros_like(rays_cam_unnorm)

        # Rotate ray directions into the robot base frame
        rays_base_dir = torch.bmm(R, rays_cam_unnorm)  # [B, 3, N]

        # Transpose to sequence format [B, N, 3]
        rays_base_dir = rays_base_dir.transpose(1, 2)
        origin_base = T.transpose(1, 2)

        # Concatenate origin and direction into a single float32 tensor [B, N, 6]
        return torch.cat([origin_base, rays_base_dir], dim=-1)


class ContinuousRayRoPE(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        # For 3D spatial encoding, head_dim must be divisible by 6
        # (3 axes * 2 for real/imaginary pairs per axis)
        if head_dim % 6 != 0:
            raise ValueError(f"ContinuousRayRoPE requires head_dim to be divisible by 6. Got {head_dim}.")

        self.dim_per_axis = head_dim // 3
        self.disable_ray_rope = False

        # Calculate inverse frequencies using the standard RoPE base formulation
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rays: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Applies Metric 3D Ray-RoPE to the input tensor.

        Args:
            x: Attention Query/Key tensor [batch_size, seq_len, num_heads, head_dim]
            rays: Tensor of shape [batch_size, seq_len, 6] representing origin and direction.
            depth_map: Metric depth prior from frozen foundation model [batch_size, 1, H, W]
        """
        if self.disable_ray_rope:
            return x

        batch_size, seq_len, num_heads, head_dim = x.shape

        with torch.amp.autocast(device_type="cuda" if x.device.type == "cuda" else "cpu", enabled=False):
            rays_f32 = rays.float()

            if depth_map is not None:
                # 1. Resize the high-res depth map to match the vision transformer's patch grid
                grid_size = int(math.sqrt(seq_len))
                depth_resized = F.interpolate(
                    depth_map.float(),
                    size=(grid_size, grid_size),
                    mode="bilinear",
                    align_corners=False
                )

                # Flatten to [B, N, 1]
                depth_flat = depth_resized.view(batch_size, 1, -1).transpose(1, 2)

                # 2. Compute absolute metric 3D point cloud coordinate: Origin + (Direction * Depth)
                if rays_f32.shape[-1] == 6:
                    origin = rays_f32[..., :3]
                    direction = rays_f32[..., 3:]
                    coords = origin + (direction * depth_flat)
                else:
                    coords = rays_f32 * depth_flat
            else:
                # Fallback to normalized directional rays if depth isn't active
                if rays_f32.shape[-1] == 6:
                    coords = F.normalize(rays_f32[..., 3:], p=2, dim=-1)
                else:
                    coords = rays_f32

            # Unbind the continuous 3D coordinates -> 3 tensors of shape [batch_size, seq_len]
            pos_x, pos_y, pos_z = coords.unbind(-1)

            inv_freq = self.inv_freq.to(x.device, dtype=torch.float32)

            freqs_x = torch.einsum("bs,d->bsd", pos_x, inv_freq)
            freqs_y = torch.einsum("bs,d->bsd", pos_y, inv_freq)
            freqs_z = torch.einsum("bs,d->bsd", pos_z, inv_freq)

            # Concatenate frequencies for all 3 axes: [batch_size, seq_len, head_dim // 2]
            freqs = torch.cat([freqs_x, freqs_y, freqs_z], dim=-1)

            # Broadcast to match num_heads: [batch_size, seq_len, 1, head_dim // 2]
            freqs = freqs.unsqueeze(2).expand(-1, -1, num_heads, -1)

            # Interleave sines and cosines
            emb = torch.cat((freqs, freqs), dim=-1)
            cos_emb = emb.cos().to(x.dtype)
            sin_emb = emb.sin().to(x.dtype)

        # Apply the rotary transformation
        x_rotated = (x * cos_emb) + (self._rotate_half(x) * sin_emb)
        return x_rotated

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)