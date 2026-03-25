from __future__ import annotations

import torch
import torch.nn as nn

from gendiff.utils.metrics import sobel_grad


class ContextualErrorModulation(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x0_hat: torch.Tensor, x_t: torch.Tensor, backprojection: torch.Tensor) -> torch.Tensor:
        intensity = torch.abs(x0_hat - x_t)
        gradient = torch.abs(sobel_grad(x0_hat) - sobel_grad(x_t))
        return self.net(torch.cat([intensity, gradient, backprojection], dim=1))
