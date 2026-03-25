from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from gendiff.models.losses import gradient_loss, image_loss, physics_loss
from gendiff.models.operator import load_operator
from gendiff.utils.metrics import summarize_metrics


class MetricTracker:
    def __init__(self) -> None:
        self.values: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, items: Dict[str, float], n: int = 1) -> None:
        for k, v in items.items():
            self.values[k] = self.values.get(k, 0.0) + float(v) * n
            self.counts[k] = self.counts.get(k, 0) + n

    def mean(self) -> Dict[str, float]:
        return {k: self.values[k] / max(self.counts[k], 1) for k in self.values}


def save_checkpoint(state: Dict, path: str | Path) -> None:
    torch.save(state, path)


def append_metrics_csv(path: str | Path, row: Dict[str, float]) -> None:
    path = Path(path)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    tracker = MetricTracker()
    for batch in tqdm(loader, desc="train", leave=False):
        ldct = batch["ldct"].to(device)
        ndct = batch["ndct"].to(device)
        sino = batch["sinogram"].to(device)
        dose = batch["dose"].to(device)
        anatomy = batch["anatomy"].to(device)
        t = torch.randint(1, cfg["model"]["num_diffusion_steps"] + 1, (ldct.shape[0],), device=device)
        xt = model.make_xt(ndct, ldct, t)
        operator = load_operator(
            cfg["operator"]["type"],
            batch["operator_path"][0] if isinstance(batch["operator_path"], list) else batch["operator_path"],
            image_shape=tuple(ndct.shape[-2:]),
            sino_shape=tuple(sino.shape[-2:]),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
            out = model.reverse_step(xt, t, dose, anatomy, sino, operator)
            li = image_loss(out.x0_hat, ndct)
            lp = physics_loss(operator.forward(out.x_prev), sino)
            lg = gradient_loss(out.x_prev, ndct)
            loss = cfg["train"]["lambda_img"] * li + cfg["train"]["lambda_phys"] * lp + cfg["train"]["lambda_grad"] * lg
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        tracker.update({"loss": loss.item(), "img": li.item(), "phys": lp.item(), "grad": lg.item()}, n=ldct.size(0))
    return tracker.mean()


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    tracker = MetricTracker()
    for batch in tqdm(loader, desc="eval", leave=False):
        ldct = batch["ldct"].to(device)
        ndct = batch["ndct"].to(device)
        sino = batch["sinogram"].to(device)
        dose = batch["dose"].to(device)
        anatomy = batch["anatomy"].to(device)
        operator = load_operator(
            cfg["operator"]["type"],
            batch["operator_path"][0] if isinstance(batch["operator_path"], list) else batch["operator_path"],
            image_shape=tuple(ndct.shape[-2:]),
            sino_shape=tuple(sino.shape[-2:]),
            device=device,
        )
        pred = model.sample(ldct, dose, anatomy, sino, operator)
        metrics = summarize_metrics(pred, ndct)
        tracker.update(metrics, n=ldct.size(0))
    return tracker.mean()
