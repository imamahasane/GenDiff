from __future__ import annotations

import torch
import torch.nn.functional as F

from gendiff.utils.metrics import sobel_grad


def dose_ranking_loss(e_d: torch.Tensor, dose: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    b = dose.shape[0]
    if b < 2:
        return e_d.new_tensor(0.0)
    direction = F.normalize(e_d.mean(dim=0, keepdim=True), dim=1)
    total = e_d.new_tensor(0.0)
    count = 0
    for i in range(b):
        for j in range(b):
            if dose[i] < dose[j]:
                proj = torch.sum((e_d[j] - e_d[i]) * direction.squeeze(0))
                total = total + F.relu(margin - (dose[j] - dose[i]) * proj)
                count += 1
    return total / max(count, 1)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    mask = mask & ~self_mask
    exp_logits = torch.exp(logits) * (~self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos.mean()


def image_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def physics_loss(forward_pred: torch.Tensor, sinogram: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(forward_pred, sinogram)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(sobel_grad(pred), sobel_grad(target))
