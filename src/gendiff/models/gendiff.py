from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import ResidualUNetBackbone
from .encoder import DoseAnatomyEncoder
from .error_modulation import ContextualErrorModulation
from .operator import BaseOperator
from .sprm import SPRM


@dataclass
class ReverseStepOutput:
    x_prev: torch.Tensor
    x0_hat: torch.Tensor
    x_phys: torch.Tensor
    error_map: torch.Tensor
    projection_residual: torch.Tensor


class GenDiff(nn.Module):
    def __init__(
        self,
        encoder: DoseAnatomyEncoder,
        backbone: ResidualUNetBackbone,
        sprm: SPRM,
        error_modulation: ContextualErrorModulation,
        num_steps: int = 20,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.backbone = backbone
        self.sprm = sprm
        self.error_modulation = error_modulation
        self.num_steps = num_steps
        self.lambda_t = nn.Parameter(torch.linspace(0.05, 0.2, num_steps))

    def make_xt(self, x0: torch.Tensor, xT: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
        alpha = (t_index.float() / self.num_steps).view(-1, 1, 1, 1)
        return alpha * x0 + (1.0 - alpha) * xT

    def reverse_step(
        self,
        x_t: torch.Tensor,
        t_index: torch.Tensor,
        dose: torch.Tensor,
        anatomy: torch.Tensor,
        sinogram: torch.Tensor,
        operator: BaseOperator,
    ) -> ReverseStepOutput:
        e_d, e_a, _, _ = self.encoder(x_t, dose, anatomy)
        zero_bp = torch.zeros_like(x_t)
        residual = self.backbone(x_t, t_index.float(), e_d, e_a, zero_bp)
        x0_hat = x_t + residual
        proj_residual = operator.forward(x0_hat) - sinogram
        backprojection = operator.adjoint(proj_residual)
        error_map = self.error_modulation(x0_hat, x_t, backprojection)
        residual = self.backbone(x_t, t_index.float(), e_d, e_a, error_map)
        x0_hat = x_t + residual
        proj_residual = operator.forward(x0_hat) - sinogram
        x_phys = x0_hat - self.lambda_t[t_index.long() - 1].view(-1, 1, 1, 1) * operator.adjoint(proj_residual)
        backprojection = operator.adjoint(proj_residual)
        error_map = self.error_modulation(x0_hat, x_t, backprojection)
        delta = self.sprm(x_phys, t_index.float(), e_d, e_a, error_map)
        x_prev = x_phys + delta
        return ReverseStepOutput(x_prev=x_prev, x0_hat=x0_hat, x_phys=x_phys, error_map=error_map, projection_residual=proj_residual)

    def sample(
        self,
        xT: torch.Tensor,
        dose: torch.Tensor,
        anatomy: torch.Tensor,
        sinogram: torch.Tensor,
        operator: BaseOperator,
    ) -> torch.Tensor:
        x = xT
        for t in range(self.num_steps, 0, -1):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            out = self.reverse_step(x, t_tensor, dose, anatomy, sinogram, operator)
            x = out.x_prev
        return x
