from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from gendiff.data.datamodule import create_dataloader
from gendiff.models.encoder import DoseAnatomyEncoder
from gendiff.models.losses import dose_ranking_loss, supervised_contrastive_loss
from gendiff.utils.config import load_config
from gendiff.utils.io import ensure_dir, save_json
from gendiff.utils.reproducibility import set_seed


def train_epoch(model, loader, optimizer, scaler, device, cfg):
    model.train()
    totals = {"loss": 0.0, "dose": 0.0, "rank": 0.0, "anat": 0.0, "cls": 0.0}
    count = 0
    for batch in tqdm(loader, desc="encoder-train", leave=False):
        x = batch["ldct"].to(device)
        dose = batch["dose"].to(device)
        anatomy = batch["anatomy"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=cfg["amp"]):
            e_d, e_a, d_pred, a_logits = model(x, dose, anatomy)
            l_dose = F.mse_loss(d_pred, dose)
            l_rank = dose_ranking_loss(e_d, dose)
            l_anat = supervised_contrastive_loss(e_a, anatomy, cfg["train"]["temperature"])
            l_cls = F.cross_entropy(a_logits, anatomy)
            loss = l_dose + cfg["train"]["lambda_rank"] * l_rank + cfg["train"]["lambda_anat"] * l_anat + 0.5 * l_cls
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        b = x.size(0)
        totals["loss"] += loss.item() * b
        totals["dose"] += l_dose.item() * b
        totals["rank"] += l_rank.item() * b
        totals["anat"] += l_anat.item() * b
        totals["cls"] += l_cls.item() * b
        count += b
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, device, cfg):
    model.eval()
    mse_total, correct, count = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="encoder-val", leave=False):
        x = batch["ldct"].to(device)
        dose = batch["dose"].to(device)
        anatomy = batch["anatomy"].to(device)
        _, _, d_pred, a_logits = model(x, dose, anatomy)
        mse_total += F.mse_loss(d_pred, dose, reduction="sum").item()
        correct += (a_logits.argmax(dim=1) == anatomy).sum().item()
        count += x.size(0)
    return {"dose_mse": mse_total / max(count, 1), "anat_acc": correct / max(count, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    run_dir = ensure_dir(Path(cfg["output_dir"]) / cfg["run_name"])
    save_json(cfg, run_dir / "config.json")

    train_loader = create_dataloader(cfg["data"]["root"], cfg["data"]["train_split"], cfg["data"]["anatomy_map"], cfg["train"]["batch_size"], cfg["num_workers"], True)
    val_loader = create_dataloader(cfg["data"]["root"], cfg["data"]["val_split"], cfg["data"]["anatomy_map"], cfg["train"]["batch_size"], cfg["num_workers"], False)

    model = DoseAnatomyEncoder(
        in_channels=cfg["model"]["in_channels"],
        base_channels=cfg["model"]["base_channels"],
        embedding_dim=cfg["model"]["embedding_dim"],
        num_anatomies=len(cfg["data"]["anatomy_map"]),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["amp"])

    best = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = train_epoch(model, train_loader, optimizer, scaler, device, cfg)
        va = val_epoch(model, val_loader, device, cfg)
        print({"epoch": epoch, **tr, **va})
        if va["dose_mse"] < best:
            best = va["dose_mse"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": cfg}, run_dir / "best_encoder.pt")


if __name__ == "__main__":
    main()
