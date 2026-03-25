from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups, num_channels, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + scale) + shift


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = AdaptiveGroupNorm(in_ch, cond_dim)
        self.norm2 = AdaptiveGroupNorm(out_ch, cond_dim)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x).view(b, c, h * w)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        head_dim = c // self.heads
        q = q.view(b, self.heads, head_dim, h * w)
        k = k.view(b, self.heads, head_dim, h * w)
        v = v.view(b, self.heads, head_dim, h * w)
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q, k) / math.sqrt(head_dim), dim=-1)
        out = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(b, c, h * w)
        out = self.proj(out).view(b, c, h, w)
        return x_in + out
