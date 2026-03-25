from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import AdaptiveGroupNorm, SinusoidalTimeEmbedding


class SPRM(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 64, embedding_dim: int = 128, time_embedding_dim: int = 128) -> None:
        super().__init__()
        cond_dim = time_embedding_dim + 2 * embedding_dim
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.norm1 = AdaptiveGroupNorm(base_channels, cond_dim)
        self.conv1 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.norm2 = AdaptiveGroupNorm(base_channels, cond_dim)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, 1, 3, padding=1)

    def forward(self, x_phys: torch.Tensor, t: torch.Tensor, e_d: torch.Tensor, e_a: torch.Tensor, error_map: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(torch.cat([self.time_embed(t), e_d, e_a], dim=1))
        h = self.conv_in(torch.cat([x_phys, error_map], dim=1))
        h = self.conv1(F.silu(self.norm1(h, cond)))
        h = self.conv2(F.silu(self.norm2(h, cond)))
        return self.conv_out(h)
