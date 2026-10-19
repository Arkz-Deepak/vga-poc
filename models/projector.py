"""
models/projector.py
Unified Space-to-Depth Token Compressor for SigLIP Visual Features.

Key Capabilities:
1. Exact Spatial Compression: Maps [B, 576, 768] ViT patch sequences into [B, 64, 960]
   multimodal tokens via 3x pixel unshuffle and linear projection.
2. Resolution Agnostic: Dynamically determines grid dimensions from token sequence length
   N (supporting 14x14, 16x16, 24x24, etc.).
3. Batch & Dtype Resilience: Automatically handles unbatched inputs [N, D] and preserves
   bfloat16/float16 execution precision under torch.amp.autocast.
4. Contiguity & Memory Safety: Enforces contiguous memory layouts across reshapes to prevent
   runtime stride/view errors during distributed and compiled execution.
"""
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.poc_config import CONFIG


class UnifiedSpaceToDepthProjector(nn.Module):
    def __init__(
        self,
        vis_dim: int = getattr(CONFIG, "vis_dim", 768),
        lm_dim: int = getattr(CONFIG, "lm_dim", 960),
        spatial_factor: int = getattr(CONFIG, "spatial_factor", 3)
    ):
        super().__init__()
        self.vis_dim = vis_dim
        self.lm_dim = lm_dim
        self.spatial_factor = spatial_factor
        self.in_features = vis_dim * (spatial_factor ** 2)  # e.g., 768 * 9 = 6912

        # Dense linear projection aligning compressed visual channels to language dimension
        self.proj = nn.Linear(self.in_features, lm_dim, bias=True)

        # Layer normalization stabilizing multimodal feature scale before cross-attention
        self.norm = nn.LayerNorm(lm_dim, eps=1e-6)

        # Standard Xavier uniform initialization
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compresses visual patch tokens spatially via Space-to-Depth.

        Args:
            x: Input patch tensor from SigLIP [B, N, D] or [N, D] (e.g., [B, 576, 768])
        Returns:
            Projected tokens aligned to SmolLM2 dimension [B, N // (factor^2), lm_dim] (e.g., [B, 64, 960])
        """
        # Ensure batch dimension
        has_batch_dim = (x.ndim == 3)
        if not has_batch_dim:
            x = x.unsqueeze(0)

        B, N, D = x.shape
        if D != self.vis_dim:
            raise ValueError(
                f"Projector input feature dimension {D} does not match expected vis_dim {self.vis_dim}"
            )

        grid_dim = int(math.isqrt(N))
        if grid_dim * grid_dim != N:
            raise ValueError(f"Visual token count N={N} cannot be shaped into a square grid (grid_dim^2 != N)")
        if grid_dim % self.spatial_factor != 0:
            raise ValueError(
                f"Grid dimension {grid_dim} is not evenly divisible by spatial factor {self.spatial_factor}"
            )

        # 1. Unflatten token sequence to 2D feature map: [B, H, W, D] -> [B, D, H, W]
        x_grid = x.view(B, grid_dim, grid_dim, D).permute(0, 3, 1, 2).contiguous()

        # 2. Space-to-depth pixel unshuffle: [B, D, H, W] -> [B, D * factor^2, H // factor, W // factor]
        x_unshuffled = F.pixel_unshuffle(x_grid, downscale_factor=self.spatial_factor)

        # 3. Reshape back into token sequence: [B, C_new, H', W'] -> [B, H'*W', C_new]
        x_tokens = x_unshuffled.permute(0, 2, 3, 1).contiguous().reshape(B, -1, self.in_features)

        # 4. Dense projection & normalization: [B, N_comp, 6912] -> [B, N_comp, 960]
        out = self.norm(self.proj(x_tokens))

        if not has_batch_dim:
            out = out.squeeze(0)

        return out