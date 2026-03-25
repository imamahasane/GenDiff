from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gendiff.data.datamodule import create_dataloader
from gendiff.models.backbone import ResidualUNetBackbone
from gendiff.models.encoder import DoseAnatomyEncoder
from gendiff.models.error_modulation import ContextualErrorModulation
from gendiff.models.gendiff import GenDiff
from gendiff.models.sprm import SPRM
from gendiff.training.engine import append_metrics_csv, evaluate, save_checkpoint, train_one_epoch
from gendiff.utils.config import load_config
from gendiff.utils.io import ensure_dir, save_json
from gendiff.utils.reproducibility import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--encoder-checkpoint", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=cfg["train"].get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    run_dir = ensure_dir(Path(cfg["output_dir"]) / cfg["run_name"])
    save_json(cfg, run_dir / "config.json")

    train_loader = create_dataloader(cfg["data"]["root"], cfg["data"]["train_split"], cfg["data"]["anatomy_map"], cfg["train"]["batch_size"], cfg["num_workers"], True)
    val_loader = create_dataloader(cfg["data"]["root"], cfg["data"]["val_split"], cfg["data"]["anatomy_map"], cfg["train"]["batch_size"], cfg["num_workers"], False)

    encoder = DoseAnatomyEncoder(
        in_channels=cfg["model"]["in_channels"],
        base_channels=32,
        embedding_dim=cfg["model"]["embedding_dim"],
        num_anatomies=len(cfg["data"]["anatomy_map"]),
    )
    if args.encoder_checkpoint:
        state = torch.load(args.encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(state["model"])
    if cfg["train"].get("freeze_encoder", False):
        for p in encoder.parameters():
            p.requires_grad = False

    backbone = ResidualUNetBackbone(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        channel_mults=cfg["model"]["channel_mults"],
        embedding_dim=cfg["model"]["embedding_dim"],
        time_embedding_dim=cfg["model"]["time_embedding_dim"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        dropout=cfg["model"]["dropout"],
    )
    sprm = SPRM(
        in_channels=2,
        base_channels=cfg["model"]["base_channels"],
        embedding_dim=cfg["model"]["embedding_dim"],
        time_embedding_dim=cfg["model"]["time_embedding_dim"],
    )
    error_modulation = ContextualErrorModulation()
    model = GenDiff(encoder, backbone, sprm, error_modulation, num_steps=cfg["model"]["num_diffusion_steps"]).to(device)

    param_groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder")], "lr": cfg["train"]["lr"]},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")], "lr": cfg["train"]["encoder_lr"]},
    ]
    optimizer = AdamW(param_groups, weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["cosine_min_lr"])
    scaler = GradScaler(enabled=cfg["amp"])

    best_metric = -float("inf") if cfg["train"]["checkpoint_mode"] == "max" else float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg)
        va = evaluate(model, val_loader, device, cfg)
        scheduler.step()
        row = {"epoch": epoch, **tr, **va}
        append_metrics_csv(run_dir / "metrics.csv", row)
        print(row)
        metric = va[cfg["train"]["checkpoint_metric"]]
        improved = metric > best_metric if cfg["train"]["checkpoint_mode"] == "max" else metric < best_metric
        if improved:
            best_metric = metric
            save_checkpoint({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "config": cfg,
            }, run_dir / "best.pt")


if __name__ == "__main__":
    main()
