# Operator-Sensitivity-DeepONets

Official repository for neural operator sensitivity analysis and analytical uncertainty quantification (UQ) using **Jacobian-DeepONet** and **Jacobian Conformal Bands (JCB)**.

This repository provides a modular, reproducible research suite for sensitivity analysis and uncertainty quantification in Deep Operator Networks (DeepONets) across four physical benchmark problems:
- **1D Burgers Equation** (advective shock fronts, steep gradient formation, and viscous dissipation)
- **2D Darcy Flow** (porous media with Gaussian Random Field permeability distributions)
- **2D Compressible Navier-Stokes** (flow-map transition operator on hyperbolic gas dynamics)
- **2D Biharmonic Equation** (fourth-order linear elliptic PDE with machine-precision linearity scaling)

---

## Methodological Framework

The research introduces two complementary paradigms for physics-informed uncertainty quantification:

### Method 1: Trained Jacobian-DeepONet (End-to-End Pinball)
- **Analytic Sensitivity Prior:** Employs the analytic branch Jacobian norm of the operator as an inductive spatial prior for uncertainty intervals, bypassing expensive stochastic ensembles.
- **Latent O(K) Gram Factorization:** In Cartesian DeepONets, trunk coordinates are isolated from input branch evaluations. The pointwise variance is evaluated in O(K) reverse-mode vector-Jacobian products (VJPs) via the latent Gram matrix $G = J_{branch} J_{branch}^T$, avoiding dense $N_{nodes} \times N_{in}$ matrix materialization.
- **Asymmetric Calibration & Implicit Sobolev Regularization:** Two learnable non-negative scalars ($w_{lower}, w_{upper}$) scale the Jacobian template through the Pinball loss function at the 90% confidence level (alpha = 0.10). By penalizing interval violations, the loss implicitly regularizes internal network derivatives to align with physical sensitivities without requiring explicit derivative supervision.

### Method 2: Jacobian Conformal Bands (JCB) (Training-Free / Post-Hoc UQ)
- **UQ for Free on Pretrained Operators:** Any pretrained Cartesian DeepONet (optimized purely for mean squared error) already contains a sensitivity geometry in its latent branch derivatives.
- **Non-Parametric Conformal Scaling:** Extracts the pointwise sensitivity template $s(x)$ in one fused VJP pass, adds a relative floor $\beta \cdot \mathrm{median}(s)$ to ensure stability at Dirichlet boundaries, and fits a single scalar nonconformity multiplier $\hat{q}$ via split-conformal prediction on held-out calibration data:
  $$y_{low/up}(x) = \mathcal{G}_\theta(u)(x) \mp \hat{q} \cdot \left( s(x) + \beta \operatorname{median}(s) \right)$$
- **Zero Additional Weights:** Requires no gradient updates and adds zero trainable parameters. Recovers the physical sensitivity correlation ($\rho_{\mathrm{Sens}}$) and tightens prediction intervals compared to standard constant-width conformal prediction.

---

## Repository Structure

- [`protocol/`](protocol/README.md): Strict protocol engine for data generation, leakage verification, job-array training, interval score checkpoint selection, and JCB post-hoc analysis.
- [`benchmarks/`](benchmarks/README.md): Evaluation suite containing post-hoc conformal estimation (`posthoc_jacobian.py`), input noise propagation, Monte Carlo benchmarks, and latency profiling.
- [`deepxde-extensions/`](deepxde-extensions/README.md): Custom neural operator architectures, including `JacobianDeepONetSoftplus`, `VanillaPinballDeepONet`, and spatial quantile heads.
- [`src/`](src/README.md): Numerical PDE solvers, dataset generators with isolated seed tracking, and procedural model training routines.

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

### 2. Method 1: Train End-to-End Jacobian-DeepONet
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

### 3. Method 2: Evaluate Jacobian Conformal Bands (JCB)
Evaluate training-free Jacobian Conformal Bands on a frozen deterministic DeepONet:
```bash
python protocol/jacobian_conformal.py darcy_small --seed 0 --beta 0.05
```
Or via the benchmark post-hoc module:
```bash
python benchmarks/posthoc_jacobian.py \
    --test data/darcy_test.h5 \
    --det modelos/darcy_mse \
    --out results/posthoc_darcy
```

### 4. Slurm HPC Pipeline
Launch the automated SLURM workflow with dependency chaining:
```bash
bash protocol/slurm/launch.sh darcy_small --smoke  # Fast smoke test
bash protocol/slurm/launch.sh darcy_small          # Full sweep
sbatch protocol/slurm/jconf.sh darcy_small         # Slurm job for JCB evaluation
```
