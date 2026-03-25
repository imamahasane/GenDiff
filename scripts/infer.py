from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from gendiff.models.operator import load_operator
from gendiff.utils.config import load_config
from gendiff.models.backbone import ResidualUNetBackbone
from gendiff.models.encoder import DoseAnatomyEncoder
from gendiff.models.error_modulation import ContextualErrorModulation
from gendiff.models.gendiff import GenDiff
from gendiff.models.sprm import SPRM


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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sample = np.load(args.input, allow_pickle=True)
    ldct = torch.from_numpy(sample["ldct"]).float().unsqueeze(0).to(device)
    sino = torch.from_numpy(sample["sinogram"]).float().unsqueeze(0).to(device)
    dose = torch.tensor([float(sample["dose"])], dtype=torch.float32, device=device)
    anatomy = sample["anatomy"].item()
    if isinstance(anatomy, bytes):
        anatomy = anatomy.decode("utf-8")
    anatomy_id = cfg["data"]["anatomy_map"][anatomy] if isinstance(anatomy, str) else int(anatomy)
    anatomy_t = torch.tensor([anatomy_id], dtype=torch.long, device=device)
    operator_path = str(sample["operator_path"]) if "operator_path" in sample else ""
    operator = load_operator(cfg["operator"]["type"], operator_path, tuple(ldct.shape[-2:]), tuple(sino.shape[-2:]), device)

    with torch.no_grad():
        pred = model.sample(ldct, dose, anatomy_t, sino, operator)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "reconstruction.npy", pred.squeeze(0).cpu().numpy())
    print(f"Saved reconstruction to {out_dir / 'reconstruction.npy'}")


if __name__ == "__main__":
    main()
