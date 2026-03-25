from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2, 3)))


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10((data_range ** 2) / mse)


def ssim_batch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    vals = []
    for p, t in zip(pred_np, target_np):
        vals.append(ssim_fn(t[0], p[0], data_range=data_range))
    return torch.tensor(vals, device=pred.device, dtype=pred.dtype)


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def summarize_metrics(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> Dict[str, float]:
    return {
        "psnr": float(psnr(pred, target, data_range).mean().item()),
        "ssim": float(ssim_batch(pred, target, data_range).mean().item()),
        "rmse": float(rmse(pred, target).mean().item()),
    }
