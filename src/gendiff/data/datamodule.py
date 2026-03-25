from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from .dataset import CTReconstructionDataset


def create_dataloader(
    root: str,
    split: str,
    anatomy_map: Dict[str, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    ds = CTReconstructionDataset(root=root, split=split, anatomy_map=anatomy_map)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )
