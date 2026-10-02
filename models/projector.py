"""
Unified Space-to-Depth Projector Module.

Role in VGA (Vision-Geometry-Action) Architecture:
-------------------------------------------------
Traditional Vision-Language models use all raw patch tokens from the vision encoder
(e.g., 576 tokens for a 24x24 patch grid from SigLIP). Feeding 576 visual tokens
into an autoregressive language model or transformer creates massive quadratic
attention overhead (O(N^2)), causing inference latency to exceed 50-100 ms and
breaking real-time 50 Hz robot control.

Solution:
---------
This module uses a hardware-friendly "Space-to-Depth" operation (pixel unshuffle):
- Groups adjacent spatial patches (e.g., 3x3 patches = 9 patches)
- Stacks their visual feature channels together into depth: 768 * 9 = 6912 dimensions
- Reduces token count from 576 (24x24) down to 64 (8x8) tokens
- Projects 6912-dim stacked tokens into language model dimension (960) via a single linear layer
- Preserves 100% of the raw visual feature information without lossy pooling or spatial dropping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedSpaceToDepthProjector(nn.Module):
    """
    Compresses high-resolution visual patch tokens into a compact grid of 64 tokens
    via pixel unshuffle followed by a linear projection to language model dimension.
    """
    def __init__(self, vis_dim: int = 768, lm_dim: int = 960, spatial_factor: int = 3):
        super().__init__()
        self.vis_dim = vis_dim
        self.lm_dim = lm_dim
        self.spatial_factor = spatial_factor
        
        # in_features = 768 * (3^2) = 768 * 9 = 6912
        # When 3x3 adjacent patches are un-shuffled into channel depth, channels multiply by factor^2
        self.in_features = vis_dim * (spatial_factor ** 2)
        
        # Projection layer: maps from 6912 -> 960 (SmolLM2 hidden dimension)
        self.proj = nn.Linear(self.in_features, lm_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual patch embeddings of shape [B, N, D]
               where B is batch size, N is number of tokens (e.g. 576 = 24x24),
               and D is visual dimension (768).
        Returns:
            Compressed visual tokens of shape [B, 64, lm_dim] (e.g. [B, 64, 960]).
        """
        B, N, D = x.shape
        grid_size = int(math.isqrt(N))
        assert grid_size * grid_size == N, f"Token count {N} must form a square grid, got grid_size={grid_size}"
        assert grid_size % self.spatial_factor == 0, (
            f"Grid size {grid_size} must be divisible by spatial_factor {self.spatial_factor}"
        )
        
        # 1. Unflatten 1D token sequence into 2D spatial grid:
        # [B, N, D] -> [B, grid_size, grid_size, D]
        x_grid = x.view(B, grid_size, grid_size, D)
        
        # 2. Permute to PyTorch standard image format [B, Channels, Height, Width]:
        # [B, grid_size, grid_size, D] -> [B, D, grid_size, grid_size]
        x_grid = x_grid.permute(0, 3, 1, 2).contiguous()
        
        # 3. Hardware-accelerated pixel unshuffle (Space-to-Depth):
        # Transposes spatial 3x3 blocks into channel depth:
        # [B, 768, 24, 24] -> [B, 768 * 9 = 6912, 8, 8]
        x_unshuffled = F.pixel_unshuffle(x_grid, downscale_factor=self.spatial_factor)
        
        # 4. Reshape back into sequence of tokens:
        # [B, 6912, 8, 8] -> [B, 8, 8, 6912] -> [B, 64, 6912]
        out_h, out_w = x_unshuffled.shape[2], x_unshuffled.shape[3]
        num_out_tokens = out_h * out_w
        x_tokens = x_unshuffled.permute(0, 2, 3, 1).contiguous().view(B, num_out_tokens, self.in_features)
        
        # 5. Linear projection from stacked visual representation into language space:
        # [B, 64, 6912] -> [B, 64, 960]
        out = self.proj(x_tokens)
        return out
