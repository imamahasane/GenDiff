from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoseAnatomyEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, embedding_dim: int = 128, num_anatomies: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        feat_dim = base_channels * 4
        self.shared = nn.Linear(feat_dim + 1 + num_anatomies, embedding_dim)
        self.to_dose_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.to_anatomy_embedding = nn.Linear(embedding_dim, embedding_dim)
        self.dose_head = nn.Linear(embedding_dim, 1)
        self.anatomy_head = nn.Linear(embedding_dim, num_anatomies)
        self.num_anatomies = num_anatomies

    def forward(self, x: torch.Tensor, dose: torch.Tensor, anatomy: torch.Tensor):
        feat = self.features(x).flatten(1)
        anatomy_onehot = F.one_hot(anatomy, num_classes=self.num_anatomies).float()
        h = torch.cat([feat, dose.unsqueeze(1), anatomy_onehot], dim=1)
        h = F.silu(self.shared(h))
        e_d = F.normalize(self.to_dose_embedding(h), dim=1)
        e_a = F.normalize(self.to_anatomy_embedding(h), dim=1)
        d_pred = self.dose_head(e_d).squeeze(1)
        a_logits = self.anatomy_head(e_a)
        return e_d, e_a, d_pred, a_logits
