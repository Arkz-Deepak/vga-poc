"""
Diffusion Transformer (DiT) Action Expert for VGA Policy.

Mathematical Foundations:
1. Continuous-Time Flow Matching:
   Instead of thousands of discrete diffusion steps, Flow Matching trains a vector field
   v_theta(x_t, t, c) to regress the straight-line probability path connecting Gaussian
   noise x_1 ~ N(0, I) to target robotic action chunk x_0 in R^{H x 7}:
       x_t = (1 - (1 - sigma_min) * t) * x_0 + t * x_1,  t in [0, 1]
       u_t(x_1 | x_0) = x_1 - (1 - sigma_min) * x_0 (target velocity)
   During training:
       L_flow = E_{t, x_0, x_1} || v_theta(x_t, t, c) - u_t ||^2

2. Adaptive Layer Normalization (AdaLN-Zero):
   Conditioning signals (diffusion time t, visual-language context c, and prefix waypoints P)
   modulate the normalization layers of each Transformer block:
       AdaLN(h, c) = (1 + gamma(c)) * LayerNorm(h) + beta(c)
       h = h + alpha(c) * Block(AdaLN(h, c))
   where gamma, beta, alpha are computed from a zero-initialized projection layer, ensuring
   the block acts as identity at initialization for stable training.

3. 4-Step Euler ODE Integration:
   For sub-18ms real-time control (>= 50 Hz), we integrate the reverse probability flow ODE:
       dx/dt = v_theta(x_t, t, c)
   using NFE = 4 uniform Euler steps from t = 1.0 to t = 0.0:
       x_{t - dt} = x_t - dt * v_theta(x_t, t, c),  where dt = 0.25
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulates normalized tensor with affine scale and shift parameters."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SinusoidalTimeEmbedding(nn.Module):
    """Encodes continuous diffusion time t in [0, 1] into frequency space."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of shape (B,) or (B, 1) in [0, 1]
        Returns:
            emb: Frequency embedding of shape (B, dim)
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class DiTBlock(nn.Module):
    """
    A single Diffusion Transformer (DiT) block equipped with AdaLN-Zero conditioning.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        # AdaLN modulation generates: scale1, shift1, gate1, scale2, shift2, gate2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # Initialize AdaLN weights to zero so blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Action token sequence [B, SeqLen, HiddenDim]
            c: Global conditioning vector [B, HiddenDim]
        Returns:
            x: Updated sequence [B, SeqLen, HiddenDim]
        """
        modulation = self.adaLN_modulation(c)
        shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)

        # 1. Self-Attention with AdaLN modulation
        norm_x1 = modulate(self.norm1(x), shift1, scale1)
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + gate1.unsqueeze(1) * attn_out

        # 2. MLP with AdaLN modulation
        norm_x2 = modulate(self.norm2(x), shift2, scale2)
        mlp_out = self.mlp(norm_x2)
        x = x + gate2.unsqueeze(1) * mlp_out

        return x


class DiTActionExpert(nn.Module):
    """
    12-layer Diffusion Transformer Action Expert with 4-step Euler ODE solver.
    Predicts 16-step action trajectories conditioned on visual-language context.
    """

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        prefix_len: int = 4,
        context_dim: int = 960,       # Matches SmolLM2 / fused context dimension
        hidden_dim: int = 384,        # Lightweight DiT hidden dimension for sub-18ms latency
        num_heads: int = 6,
        num_layers: int = 12,
        euler_steps: int = 4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.prefix_len = prefix_len
        self.hidden_dim = hidden_dim
        self.euler_steps = euler_steps

        # 1. Input Action Embeddings (Linear projection of 7D action vectors)
        self.action_in_proj = nn.Linear(action_dim, hidden_dim)
        # Position embedding for the 16 action steps
        self.pos_emb = nn.Parameter(torch.zeros(1, action_horizon, hidden_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 2. Conditioning MLPs
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Optional prefix waypoint projection (P=4 steps from previous chunk)
        self.prefix_proj = nn.Linear(action_dim, hidden_dim)

        # 3. Stack of 12 DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        # 4. Final Output Projection to Velocity Field v_theta in R^{B x 16 x 7}
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.action_out_proj = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        prefix_waypoints: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predicts velocity field v_theta(x_t, t, context).

        Args:
            x_t: Noisy action sequence [B, 16, 7]
            t: Diffusion timestep tensor [B] in [0, 1]
            context: Condition vector [B, context_dim] (or pooled visual-language tokens)
            prefix_waypoints: Optional tail of previous chunk [B, 4, 7]
        Returns:
            v_theta: Velocity prediction [B, 16, 7]
        """
        # 1. Project actions and add learned positional embeddings
        h = self.action_in_proj(x_t) + self.pos_emb  # [B, 16, hidden_dim]

        # 2. Combine time and context conditioning into single conditioning vector c
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        c_emb = self.context_proj(context)  # [B, hidden_dim]
        c = t_emb + c_emb  # [B, hidden_dim]

        # 3. If prefix waypoints provided, inject their summary into conditioning
        if prefix_waypoints is not None:
            p_emb = self.prefix_proj(prefix_waypoints).mean(dim=1)  # [B, hidden_dim]
            c = c + p_emb

        # 4. Pass through 12 DiT Transformer Blocks
        for block in self.blocks:
            h = block(h, c)

        # 5. Final AdaLN modulation and linear projection to velocity
        final_shift, final_scale = self.final_adaLN(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), final_shift, final_scale)
        v_pred = self.action_out_proj(h)  # [B, 16, 7]

        return v_pred

    @torch.no_grad()
    def sample_actions(
        self,
        context: torch.Tensor,
        prefix_waypoints: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generates 16-step action trajectories using 4-step Euler ODE integration.
        Operates from t = 1.0 (pure Gaussian noise) to t = 0.0 (clean action chunk).

        Args:
            context: Condition vector [B, context_dim]
            prefix_waypoints: Optional tail of previous chunk [B, 4, 7]
            generator: Optional torch random generator
        Returns:
            actions: Predicted action chunk [B, 16, 7]
        """
        batch_size = context.shape[0]
        device = context.device

        # 1. Sample initial Gaussian noise x_1 ~ N(0, I)
        x = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            generator=generator,
            device=device,
            dtype=context.dtype,
        )

        # 2. Uniform Euler integration over NFE = 4 steps
        steps = self.euler_steps
        dt = 1.0 / steps
        timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for step_idx in range(steps):
            t_curr = timesteps[step_idx].expand(batch_size)
            # Evaluate vector field v_theta at current state and time
            v = self.forward(x, t_curr, context, prefix_waypoints=prefix_waypoints)
            # Euler ODE step: x_{t - dt} = x_t - dt * v
            x = x - dt * v

        return x
