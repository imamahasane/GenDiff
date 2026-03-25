from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse
import torch


class BaseOperator:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityOperator(BaseOperator):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y


class MatrixOperator(BaseOperator):
    def __init__(self, matrix: torch.Tensor, image_shape: tuple[int, int], sino_shape: tuple[int, int]) -> None:
        self.matrix = matrix
        self.image_shape = image_shape
        self.sino_shape = sino_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.view(b, -1)
        y = flat @ self.matrix.T
        return y.view(b, 1, *self.sino_shape)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        b = y.shape[0]
        flat = y.view(b, -1)
        x = flat @ self.matrix
        return x.view(b, 1, *self.image_shape)


class SparseMatrixOperator(BaseOperator):
    def __init__(self, matrix: torch.Tensor, image_shape: tuple[int, int], sino_shape: tuple[int, int]) -> None:
        self.matrix = matrix.coalesce()
        self.matrix_t = torch.transpose(self.matrix, 0, 1).coalesce()
        self.image_shape = image_shape
        self.sino_shape = sino_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for flat in x.view(x.shape[0], -1):
            outs.append(torch.sparse.mm(self.matrix, flat[:, None]).squeeze(1))
        return torch.stack(outs, dim=0).view(x.shape[0], 1, *self.sino_shape)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        outs = []
        for flat in y.view(y.shape[0], -1):
            outs.append(torch.sparse.mm(self.matrix_t, flat[:, None]).squeeze(1))
        return torch.stack(outs, dim=0).view(y.shape[0], 1, *self.image_shape)


def load_operator(operator_type: str, operator_path: str, image_shape: tuple[int, int], sino_shape: tuple[int, int], device: torch.device):
    if operator_type == "identity" or not operator_path:
        return IdentityOperator()
    path = Path(operator_path)
    if operator_type == "matrix":
        if path.suffix == ".npy":
            matrix = torch.from_numpy(np.load(path)).float().to(device)
        elif path.suffix == ".pt":
            matrix = torch.load(path, map_location=device).float()
        else:
            raise ValueError(f"Unsupported dense operator file: {path}")
        return MatrixOperator(matrix, image_shape, sino_shape)
    if operator_type == "sparse_matrix":
        sparse = scipy.sparse.load_npz(path)
        coo = sparse.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data).float()
        matrix = torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device)
        return SparseMatrixOperator(matrix, image_shape, sino_shape)
    raise ValueError(f"Unknown operator type: {operator_type}")
