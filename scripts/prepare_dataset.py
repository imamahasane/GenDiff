from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize_hu(x: np.ndarray, hu_min: float = -1024.0, hu_max: float = 1024.0) -> np.ndarray:
    x = np.clip(x, hu_min, hu_max)
    x = 2.0 * (x - hu_min) / (hu_max - hu_min) - 1.0
    return x.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert raw per-slice arrays into the GenDiff .npz format.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hu-min", type=float, default=-1024.0)
    parser.add_argument("--hu-max", type=float, default=1024.0)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = input_root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Expected metadata.json in input root.")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    for split, items in metadata.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            ldct = normalize_hu(np.load(input_root / item["ldct"]), args.hu_min, args.hu_max)[None, ...]
            ndct = normalize_hu(np.load(input_root / item["ndct"]), args.hu_min, args.hu_max)[None, ...]
            sinogram = np.load(input_root / item["sinogram"]).astype(np.float32)
            out_path = split_dir / f"{item['id']}.npz"
            np.savez_compressed(
                out_path,
                ldct=ldct,
                ndct=ndct,
                sinogram=sinogram,
                dose=np.float32(item["dose"]),
                anatomy=np.array(item["anatomy"]),
                operator_path=np.array(item.get("operator_path", "")),
            )
    print(f"Prepared dataset at {output_root}")


if __name__ == "__main__":
    main()
