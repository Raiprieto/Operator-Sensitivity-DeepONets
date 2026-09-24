# Trained Checkpoints and Model Zoo

This directory contains the verified, audited model checkpoints backing the paper results (Table 1 `tab:main`, Table 2 `tab:notrain`, and Table 3 `tab:fair_quantile`). All models share the compact backbone ($5 \times 100$, latent dimension $K=100$) evaluated over three random seeds ($n=3$).

---

## Directory Organization

The models are categorized hierarchically by physical partial differential equation, uncertainty quantification variant, and seed:

```
models/
├── burgers/
│   ├── deterministic/          # Pure MSE baseline (no bands)
│   │   ├── seed0/
│   │   ├── seed1/
│   │   └── seed2/
│   ├── constant_width/         # Vanilla Pinball baseline (constant width)
│   │   ├── seed0/
│   │   ├── seed1/
│   │   └── seed2/
│   ├── spatial_quantile/       # Spatial Pinball quantile baseline (quantile_x)
│   │   ├── seed0/
│   │   ├── seed1/
│   │   └── seed2/
│   ├── conditional_quantile/   # Conditional Quantile baseline (quantile_ux / qux)
│   │   ├── seed0/
│   │   ├── seed1/
│   │   └── seed2/
│   ├── jacobian/               # Jacobian-DeepONet (VJP + Pinball loss, EMA scale)
│   │   ├── seed0/
│   │   ├── seed1/
│   │   └── seed2/
│   └── benchmark_summaries/    # Summary metrics across seeds
├── darcy/
│   ├── deterministic/          # Pure MSE baseline (Table 2 JCB reference)
│   ├── constant_width/         # Vanilla Pinball
│   ├── spatial_quantile/       # Spatial quantile head
│   ├── conditional_quantile/   # Sample-conditioned quantile head
│   ├── jacobian/               # Jacobian-DeepONet (lambda=4)
│   └── benchmark_summaries/
└── navier_stokes/
    ├── deterministic/          # Pure MSE baseline (Table 2 JCB reference)
    ├── constant_width/         # Vanilla Pinball
    ├── spatial_quantile/       # Spatial quantile head
    ├── conditional_quantile/   # Sample-conditioned quantile head
    ├── jacobian/               # Jacobian-DeepONet (lambda=4)
    └── benchmark_summaries/
```

---

## Contents of Each Seed Folder

Each individual seed directory contains:
- `best.pt`: PyTorch model weights selected by validation interval score during training.
- `scalers.npz`: Coordinate and field normalization scalers (`u_mean`, `u_std`, `y_mean`, `y_std`).
- `metadata.json`: Full training configuration, architecture hyperparameters, and fixed reference uncertainty (`unc_ref`).
- `val_history.csv` / `loss_history.csv`: Training and validation metric progression.
- `test_metrics.json`: Uncalibrated test metrics (physical MSE, raw PICP, raw MPIW, interval score).
- `test_correlations.json`: Spearman rank correlations ($\rho_{\text{Err}}$, $\rho_{\text{Sens}}$).
- `conformal.json`: Post-hoc split-conformal multiplier $\hat{q}$ and calibrated width at 90% nominal coverage.
- `jacobian_conformal.json`: Multi-shape conformal benchmark metrics (const_phys, const_norm, jac, jac_log).

---

## Python Loading Example

You can inspect and load any checkpoint with Python using the protocol loader:

```python
import json
import numpy as np
import torch
import sys
sys.path.insert(0, "protocol")
import common as C

# Path to the desired model run
model_dir = "models/darcy/jacobian/seed0"

# Load architecture and hyperparameters directly from metadata
meta = json.load(open(f"{model_dir}/metadata.json"))
net = C.build_net(meta["architecture"], meta["model"], meta["sigma"]).cuda()

# Load weights and freeze fixed reference scale
checkpoint = torch.load(f"{model_dir}/best.pt", map_location="cuda")
net.load_state_dict(checkpoint["model_state_dict"])
net.eval()
if "unc_ref" in meta:
    net.unc_ref = meta["unc_ref"]

# Load dataset normalization scalers
scalers = dict(np.load(f"{model_dir}/scalers.npz"))
print(f"Loaded {meta['model']} model successfully from {model_dir}")
```
