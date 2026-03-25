# Dataset format

GenDiff is implemented as a slice-level pipeline.

## Recommended processed layout

```text
processed_data/
├── train/
│   ├── sample_000001.npz
│   ├── sample_000002.npz
│   └── ...
├── val/
├── test/
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Each `.npz` file should contain:

- `ldct`: float32 array `[1, H, W]`
- `ndct`: float32 array `[1, H, W]`
- `sinogram`: float32 array `[1, M, N]` or `[M, N]`
- `dose`: float32 scalar, for example `0.10`, `0.25`, `0.50`
- `anatomy`: string encoded as bytes or integer id
- optional `operator_path`: path to a `.npz` sparse operator matrix or `.pt` tensor operator
- optional `meta`: JSON string or auxiliary metadata

## Operator choices

The code supports three operator modes:

1. `identity`: useful for debugging or image-domain denoising studies
2. `matrix`: uses an explicit projection matrix `A`
3. `sparse_matrix`: uses a sparse matrix for large-scale CT systems

For faithful physics-consistent reconstruction, provide an operator consistent with how the sinograms were generated.

## Intensity normalization

Recommended default:

- clip HU to `[-1024, 1024]`
- normalize to `[-1, 1]`

These values are configurable and should match your preprocessing used in training and evaluation.
