# GenDiff: Dose-Aware Cold Diffusion with Physics Consistency for Generalizable Low-Dose CT Reconstruction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

Official implementation of the paper:

> **“GenDiff: Dose-Aware Cold Diffusion with Physics Consistency for Generalizable Low-Dose CT Reconstruction”**  
> *Md Imam Ahasan, Guangchao Yang, A. F. M. Abdun Noor, Kah Ong Michael Goh, S. M. Hasan Mahmud, Md Mahfuzur Rahman*  
> Submitted to **PeerJ Computer Science (Applications of  AI category)**, March 2026.

---

## 1. Project Descriptionz
Low-dose CT (LDCT) imaging is essential for reducing radiation exposure but introduces significant noise and structural degradation. While deep learning methods have shown promising results, most approaches struggle with generalization across dose levels and fail to incorporate underlying imaging physics. We propose GenDiff, a dose-aware cold diffusion framework with explicit physics-consistency constraints, enabling robust reconstruction across diverse clinical conditions.

**Core features :**
- Efficient Cold Diffusion Framework
- Dose-Anatomy Encoder (DAE)
- Physics-Consistency Integration
- Iterative Refinement with Learned Priors
- Context-Aware Error Modulation (CEM)
- Stochastic Prior Refinement Module (SPRM)
- Cross-Dose & Cross-Anatomy Generalization

---

## 2. Dataset Information
This project is designed for LDCT reconstruction and supports multiple public datasets commonly used in medical imaging research. The framework is flexible and can be extended to other CT datasets with similar data formats.

**LoDoPaB-CT Dataset :**
- **Reference:** Leuschner et al., *Scientific Data* 8, 109 (2021) *LoDoPaB-CT, a benchmark dataset for low-dose computed tomography reconstruction.*
- **DOI:** [https://doi.org/10.1038/s41597-021-00893-z](https://doi.org/10.1038/s41597-021-00893-z)
- **Website:** [https://zenodo.org/records/3384092](https://zenodo.org/records/3384092)
- **Description:** Synthetic LDCT dataset derived from the LIDC-IDRI thoracic CT collection.  
- **Usage here:**  
  - 40 000 training, 3 500 validation, 3 500 test slices  
  - Sinogram → NDCT image pairs  
  - 362×362 → 256×256 (resized)  
  - Normalized to [0, 1]

**NIH–AAPM–Mayo Low-Dose CT Dataset :**
- **Reference:** Moen et al., *Medical Physics* 48(2):902–911 (2021) *Low-dose CT image and projection dataset.*
- **DOI:** [https://doi.org/10.1002/mp.14594](https://doi.org/10.1002/mp.14594)
- **Website:** [https://www.aapm.org/grandchallenge/lowdosect/  ](https://www.aapm.org/grandchallenge/lowdosect/)
- **Description:** Abdominal CT volumes with both normal-dose (NDCT) and simulated low-dose (LDCT) images.  
- **Usage here:**  
  - 8 patients for training (~4 800 slices)  
  - 2 patients for testing (~1 100 slices)  
  - Resized to 256×256, normalized to [0, 1]
  
---

## 3. Code Information
The repository is designed with a modular, scalable, and research-oriented structure, enabling easy experimentation, reproducibility, and extension.
**Repository layout :**
```bash
gendiff/
├── configs/                    
│   ├── train_stage1.yaml
│   ├── train_stage2.yaml
│   ├── eval.yaml
│   └── inference.yaml
│
├── data/                       
│   ├── dataset.py             
│   ├── transforms.py          
│   └── prepare_data.py        
│
├── models/                    
│   ├── dae.py                 
│   ├── backbone.py            
│   ├── cem.py                 
│   ├── sprm.py                
│   └── blocks.py              
│
├── diffusion/                 
│   ├── scheduler.py           
│   ├── process.py             
│   └── sampler.py             
│
├── physics/                   
│   ├── ct_operator.py         
│   ├── projector.py           
│   └── consistency.py         
│
├── losses/                    
│   ├── reconstruction.py      
│   ├── perceptual.py          
│   ├── diffusion_loss.py      
│   └── total_loss.py          
│
├── trainers/                  
│   ├── trainer_stage1.py      
│   ├── trainer_stage2.py      
│   └── base_trainer.py        
│
├── evaluation/                
│   ├── metrics.py             
│   └── evaluator.py           
│
├── utils/                     
│   ├── logger.py              
│   ├── seed.py                
│   ├── checkpoint.py          
│   └── config.py              
│
├── scripts/                   
│   ├── prepare_data.py        
│   └── visualize.py           
│
├── experiments/               
│   ├── logs/
│   ├── checkpoints/
│   └── outputs/
│
├── train_stage1.py            
├── train_stage2.py            
├── evaluate.py                
├── inference.py               
│
├── requirements.txt           
├── README.md                  
└── LICENSE                    
```

---

## 4. Method summary

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

---

## 5. Installation

```bash
conda create -n gendiff python=3.10 -y
conda activate gendiff
pip install -r requirements.txt
pip install -e .
```

---

## 6. Dataset preparation

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

---

## 7. Training

### Stage 1: Pretrain the Dose-Anatomy Encoder

```bash
python scripts/train_encoder.py --config configs/encoder_pretrain.yaml
```

### Stage 2: Train GenDiff

```bash
python scripts/train_gendiff.py --config configs/gendiff_train.yaml
```

---

## 8. Evaluation

```bash
python scripts/evaluate.py \
  --config configs/gendiff_train.yaml \
  --checkpoint runs/gendiff/best.pt \
  --split test
```

---

## 9. Inference

```bash
python scripts/infer.py \
  --config configs/gendiff_train.yaml \
  --checkpoint runs/gendiff/best.pt \
  --input /path/to/sample.npz \
  --output /path/to/output_dir
```

---

## 10. Reproducibility checklist

- global seed control
- deterministic cuDNN flags where possible
- serialized configs saved with every run
- checkpointing of optimizer, scheduler, scaler, and RNG-sensitive state
- exact train/val/test split file support
- CSV and JSON metrics export
- experiment directory versioning

---

## 11. Notes on faithful reproduction

The manuscript specifies the high-level method clearly, including the two-stage training scheme, the reverse-step equations, the datasets, and evaluation protocol. However, some exact engineering values are not fully specified in the paper, including:

- exact encoder and backbone widths
- exact attention block design
- exact `w1, w2, w3` values
- exact cold diffusion degradation schedule coefficients
- exact value ranges and normalization convention for CT intensities

This repository therefore uses **explicit, configurable defaults** chosen to remain faithful to the method while keeping the code complete and executable. To reproduce the reported numbers as closely as possible, you should set these values from the authors' final training logs if available.

---

## 12. License & Contributions

**License :**
Released under the MIT License. © 2026 GenDiff Authors. All rights reserved.

**Contribution Guidelines :**
We welcome pull requests and improvements.

---

## 13. Contact

1. Md Imam Ahasan - L2300448@stu.cqu.edu.cn
2. Guangchao Yang - gchao_yang@cqu.edu.cn
3. A. F. M Abdun Noor - 252-44-010@diu.edu.bd

---

## 14. Acknowledgements
This work was conducted at the **College of Computer Science, Chongqing University** and the **Department of Software Engineering, Daffodil International University.**
All scientific content, data processing, and results were **independently verified and approved** by the authors.
