# Operator-Sensitivity-DeepONets

Official repository for neural operator sensitivity analysis and analytical uncertainty quantification (UQ) using **Jacobian-DeepONet**.

This repository provides a modular, reproducible research suite for sensitivity and uncertainty quantification in Deep Operator Networks (DeepONets) across four physical benchmark problems:
- **1D Burgers Equation** (advective shock fronts, steep gradient formation, and viscous dissipation)
- **2D Darcy Flow** (porous media with Gaussian Random Field permeability distributions)
- **2D Compressible Navier-Stokes** (flow-map transition operator on hyperbolic gas dynamics)
- **2D Biharmonic Equation** (fourth-order linear elliptic PDE with machine-precision linearity scaling)

---

## Methodological Overview

1. **Analytical Sensitivity as a Spatial Shape Prior:** Rather than training high-capacity probabilistic quantile heads or running computationally expensive Monte Carlo ensembles, we leverage the analytic Jacobian norm of the branch network as a physically grounded spatial shape prior for prediction intervals.
2. **Latent O(K) Gram Matrix Factorization:** For Cartesian DeepONet architectures, the trunk network processes only spatial coordinates and is functionally independent of branch sensor evaluations. Exploiting this algebraic structure allows exact pointwise variance computation via the latent Gram matrix in O(K) reverse-mode vector-Jacobian products, bypassing GPU out-of-memory limitations.
3. **End-to-End Asymmetric Pinball Calibration:** Two learnable non-negative scalars (w_lower, w_upper) scale the spatial uncertainty shape via the Pinball loss function at the 90% confidence level (alpha = 0.10). This mechanism acts as an implicit Sobolev regularizer: model sensitivities naturally align with the underlying physical gradients without explicit derivative supervision during training.
4. **Split-Conformal Calibration:** Post-hoc conformal rescaling on held-out calibration splits guarantees target empirical coverage, enabling fair and direct comparison of interval widths across baseline models.

---

## Repository Structure

- [`protocol/`](protocol/README.md): Strict protocol engine for data generation, leakage verification, job-array training, interval score checkpoint selection, and aggregation.
- [`benchmarks/`](benchmarks/README.md): Evaluation suite containing post-hoc conformal estimation on deterministic models, input noise propagation, Monte Carlo benchmarks, and latency profiling.
- [`deepxde-extensions/`](deepxde-extensions/README.md): Custom neural operator architectures, including `JacobianDeepONet`, `VanillaPinballDeepONet`, and spatial quantile heads.
- [`src/`](src/README.md): Ground-truth PDE solvers, data generation routines with strict seed control, and procedural model training pipelines.

---

## Installation and Requirements

A Python 3.9+ virtual environment is recommended:

```bash
python -m venv env_tesis
source env_tesis/bin/activate  # On Windows: env_tesis\Scripts\activate
pip install -r requirements.txt
```

Core dependencies include PyTorch, DeepXDE, JAX, NumPy, SciPy, Matplotlib, and H5PY.

---

## Quickstart

### 1. Data Generation
Generate certified dataset splits with isolated seed offsets:
```bash
python src/data_generation/darcy_pdebench.py --num_samples 1000 --N 64 --output_path data/darcy_test.h5 --seed_offset 1000000
```

### 2. Model Training
Train a Jacobian-DeepONet with asymmetric Pinball loss and latent O(K) variance:
```bash
python src/model_training/deeponet_training.py \
    --data_path data/darcy_train.h5 \
    --output_path modelos/darcy_jacobian \
    --use_cartesian_prod \
    --use_jacobian \
    --variance_activation softplus \
    --pinball_lambda 4.0 \
    --iterations 200000 \
    --batch_size 128
```

### 3. Slurm HPC Protocol
Launch the automated SLURM pipeline on compute nodes with afterok dependencies:
```bash
bash protocol/slurm/launch.sh darcy_small --smoke  # Fast smoke test
bash protocol/slurm/launch.sh darcy_small          # Full parameter sweep
```
