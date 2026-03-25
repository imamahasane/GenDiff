from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ResidualBlock, SelfAttention2d, SinusoidalTimeEmbedding


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class ResidualUNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] | tuple[int, ...] = (1, 2, 4, 8),
        embedding_dim: int = 128,
        time_embedding_dim: int = 128,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels + 1, base_channels, 3, padding=1)
        self.time_embed = SinusoidalTimeEmbedding(time_embedding_dim)
        cond_dim = time_embedding_dim + 2 * embedding_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        downs = []
        ch = base_channels
        skips = []
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                downs.append(ResidualBlock(ch, out_ch, cond_dim, dropout))
                ch = out_ch
                if mult >= 4:
                    downs.append(SelfAttention2d(ch))
                skips.append(ch)
            if i != len(channel_mults) - 1:
                downs.append(Downsample(ch))
                skips.append(ch)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResidualBlock(ch, ch, cond_dim, dropout),
            SelfAttention2d(ch),
            ResidualBlock(ch, ch, cond_dim, dropout),
        ])

        ups = []
        for i, mult in list(enumerate(channel_mults))[::-1]:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = skips.pop()
                ups.append(ResidualBlock(ch + skip_ch, out_ch, cond_dim, dropout))
                ch = out_ch
                if mult >= 4:
                    ups.append(SelfAttention2d(ch))
            if i != 0:
                ups.append(Upsample(ch))
        self.ups = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def make_condition(self, t: torch.Tensor, e_d: torch.Tensor, e_a: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        cond = torch.cat([t_emb, e_d, e_a], dim=1)
        return self.cond_proj(cond)

    def forward(self, x: torch.Tensor, t: torch.Tensor, e_d: torch.Tensor, e_a: torch.Tensor, error_map: torch.Tensor) -> torch.Tensor:
        cond = self.make_condition(t, e_d, e_a)
        x = self.in_conv(torch.cat([x, error_map], dim=1))
        skip_feats = []
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
            skip_feats.append(x)
        for layer in self.mid:
            if isinstance(layer, ResidualBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                skip = skip_feats.pop()
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                x = torch.cat([x, skip], dim=1)
                x = layer(x, cond)
            else:
                x = layer(x)
        return self.out_conv(F.silu(self.out_norm(x)))
