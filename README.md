# GenDiff

A reproducible PyTorch implementation of **GenDiff: A dose and anatomy aware diffusion model with structural prior refinement for low-dose CT reconstruction and generalization**.

This repository implements the method described in the manuscript, including:

- Dose-Anatomy Encoder with stage-1 pretraining
- Dose/anatomy conditioned cold diffusion backbone
- Physics-consistency update using a configurable CT operator
- Contextual Error Modulation
- Structural Prior Refinement Module (SPRM)
- Stage-2 joint optimization with image, physics, and gradient losses
- Deterministic training controls, configuration-driven experiments, logging, checkpointing, evaluation, and inference

The implementation follows the manuscript closely. Some low-level design details are not fully specified in the paper (for example exact channel widths, some normalization choices, and exact cold diffusion schedule), so the repository provides **paper-aligned defaults** that are fully documented and easy to modify.

## Paper-to-code mapping

| Paper component | Code |
|---|---|
| Dose-Anatomy Encoder | `src/gendiff/models/encoder.py` |
| Cold Diffusion Backbone | `src/gendiff/models/backbone.py` |
| Contextual Error Modulation | `src/gendiff/models/error_modulation.py` |
| Structural Prior Refinement Module (SPRM) | `src/gendiff/models/sprm.py` |
| Physics-consistency update | `src/gendiff/models/operator.py`, `src/gendiff/models/gendiff.py` |
| Stage-1 encoder pretraining | `scripts/train_encoder.py` |
| Stage-2 joint training | `scripts/train_gendiff.py` |
| Evaluation / inference | `scripts/evaluate.py`, `scripts/infer.py` |

## Method summary

GenDiff formulates LDCT reconstruction as a **deterministic cold diffusion process**. Starting from an LDCT slice `x_T`, the model iteratively refines the sample over `T` reverse steps.

At each step `t`:

1. The diffusion backbone predicts a residual `r_t = f_theta(x_t, t, e_d, e_a)`.
2. A tentative clean image is computed as `x0_hat = x_t + r_t`.
3. A physics-consistency correction is applied: `x_phys = x0_hat - lambda_t A^T(A x0_hat - y)`.
4. A contextual error map is computed from image and projection inconsistencies.
5. SPRM predicts a refinement term `Delta x_t`.
6. The next state is `x_{t-1} = x_phys + Delta x_t`.

The stage-2 objective follows the manuscript:

- image fidelity loss
- physics consistency loss
- gradient consistency loss

## Repository structure

```text
GenDiff/
├── configs/
│   ├── encoder_pretrain.yaml
│   └── gendiff_train.yaml
├── docs/
│   └── dataset_format.md
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_encoder.py
│   ├── train_gendiff.py
│   ├── evaluate.py
│   └── infer.py
├── src/gendiff/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── requirements.txt
└── setup.py
```

## Installation

```bash
conda create -n gendiff python=3.10 -y
conda activate gendiff
pip install -r requirements.txt
pip install -e .
```

## Dataset preparation

Expected slice-level format is documented in [`docs/dataset_format.md`](docs/dataset_format.md).

Each sample should provide at minimum:

- `ldct`: low-dose reconstructed slice, shape `[1, H, W]`
- `ndct`: reference high-dose slice, shape `[1, H, W]`
- `sinogram`: measured or simulated projection data
- `dose`: scalar float in `(0, 1]`
- `anatomy`: one of `abdomen`, `chest`, `head`
- optional operator metadata or path to an operator matrix

A preprocessing utility is provided:

```bash
python scripts/prepare_dataset.py \
  --input-root /path/to/raw_data \
  --output-root /path/to/processed_data
```

## Training

### Stage 1: Pretrain the Dose-Anatomy Encoder

```bash
python scripts/train_encoder.py --config configs/encoder_pretrain.yaml
```

### Stage 2: Train GenDiff

```bash
python scripts/train_gendiff.py --config configs/gendiff_train.yaml
```

## Evaluation

```bash
python scripts/evaluate.py \
  --config configs/gendiff_train.yaml \
  --checkpoint runs/gendiff/best.pt \
  --split test
```

## Inference

```bash
python scripts/infer.py \
  --config configs/gendiff_train.yaml \
  --checkpoint runs/gendiff/best.pt \
  --input /path/to/sample.npz \
  --output /path/to/output_dir
```

## Reproducibility checklist

- global seed control
- deterministic cuDNN flags where possible
- serialized configs saved with every run
- checkpointing of optimizer, scheduler, scaler, and RNG-sensitive state
- exact train/val/test split file support
- CSV and JSON metrics export
- experiment directory versioning

## Notes on faithful reproduction

The manuscript specifies the high-level method clearly, including the two-stage training scheme, the reverse-step equations, the datasets, and evaluation protocol. However, some exact engineering values are not fully specified in the paper, including:

- exact encoder and backbone widths
- exact attention block design
- exact `w1, w2, w3` values
- exact cold diffusion degradation schedule coefficients
- exact value ranges and normalization convention for CT intensities

This repository therefore uses **explicit, configurable defaults** chosen to remain faithful to the method while keeping the code complete and executable. To reproduce the reported numbers as closely as possible, you should set these values from the authors' final training logs if available.

## Citation

Please cite the original manuscript when using this repository.
