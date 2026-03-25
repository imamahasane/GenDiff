from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gendiff.data.datamodule import create_dataloader
from gendiff.models.backbone import ResidualUNetBackbone
from gendiff.models.encoder import DoseAnatomyEncoder
from gendiff.models.error_modulation import ContextualErrorModulation
from gendiff.models.gendiff import GenDiff
from gendiff.models.sprm import SPRM
from gendiff.training.engine import evaluate
from gendiff.utils.config import load_config
from gendiff.utils.io import save_json


def build_model(cfg):
    encoder = DoseAnatomyEncoder(1, 32, cfg["model"]["embedding_dim"], len(cfg["data"]["anatomy_map"]))
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
    sprm = SPRM(2, cfg["model"]["base_channels"], cfg["model"]["embedding_dim"], cfg["model"]["time_embedding_dim"])
    return GenDiff(encoder, backbone, sprm, ContextualErrorModulation(), cfg["model"]["num_diffusion_steps"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    split_name = cfg["data"].get(f"{args.split}_split", args.split)
    loader = create_dataloader(cfg["data"]["root"], split_name, cfg["data"]["anatomy_map"], cfg["train"]["batch_size"], cfg["num_workers"], False)
    metrics = evaluate(model, loader, device, cfg)
    print(metrics)
    save_json(metrics, Path(args.checkpoint).with_name(f"metrics_{args.split}.json"))


if __name__ == "__main__":
    main()
