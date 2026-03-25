from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class CTReconstructionDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        anatomy_map: Dict[str, int],
    ) -> None:
        self.root = Path(root) / split
        self.files: List[Path] = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")
        self.anatomy_map = anatomy_map

    def __len__(self) -> int:
        return len(self.files)

    def _decode_anatomy(self, value: Any) -> int:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, np.ndarray):
            value = value.item()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
        if isinstance(value, (int, np.integer)):
            return int(value)
        if value not in self.anatomy_map:
            raise KeyError(f"Unknown anatomy label: {value}")
        return self.anatomy_map[value]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.files[index]
        arr = np.load(path, allow_pickle=True)
        ldct = torch.from_numpy(arr["ldct"]).float()
        ndct = torch.from_numpy(arr["ndct"]).float()
        sino = torch.from_numpy(arr["sinogram"]).float()
        dose = torch.tensor(float(arr["dose"]), dtype=torch.float32)
        anatomy = torch.tensor(self._decode_anatomy(arr["anatomy"]), dtype=torch.long)
        operator_path = str(arr["operator_path"]) if "operator_path" in arr else ""
        return {
            "ldct": ldct,
            "ndct": ndct,
            "sinogram": sino,
            "dose": dose,
            "anatomy": anatomy,
            "operator_path": operator_path,
            "id": path.stem,
        }
