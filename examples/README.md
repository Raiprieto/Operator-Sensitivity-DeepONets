# Reproducible Workflow Examples

This directory provides concrete, executable bash scripts illustrating data generation, model training, and post-training evaluation across all physical benchmark equations and model variants.

---

## Directory Organization

```
examples/
├── data_generation/      # Certified dataset generation with seed isolation
├── training/             # Model training across equations and architectures
└── testing/              # Testing suites: JCB, uniform evaluation, noise, latency
```

---

## 1. Data Generation Examples (`examples/data_generation/`)

Each script demonstrates generating non-overlapping training and test splits with certified seed offsets:

- **`generate_burgers.sh`**: Generates 1D Burgers training data (omitting Jacobians) and test data with physical sensitivity derivatives ($d u / d \nu$).
- **`generate_darcy.sh`**: Generates 2D Darcy Flow datasets with Gaussian Random Field permeability. Illustrates in-distribution generation ($\mathrm{scale}=0.5$) and out-of-distribution shifts ($\mathrm{scale}=0.75$).
- **`generate_navier_stokes.sh`**: Generates 2D Compressible Navier-Stokes Flow Map data ($t=0 \to t=0.5$), including reference sensitivities ($d \rho / d \Gamma$).
- **`generate_biharmonic.sh`**: Generates disjoint 2D Biharmonic test sets using machine-precision linearity scaling ($f = c \cdot g, u = c \cdot u_{\mathrm{base}}$).

Run from the repository root:
```bash
bash examples/data_generation/generate_darcy.sh
```

---

## 2. Model Training Examples (`examples/training/`)

Each script trains candidate architectures on the specified PDE using the compact, standardized hyperparameter recipe:

- **`train_burgers.sh`**: Trains Jacobian-DeepONet ($\lambda=4.0$), Vanilla Pinball, Conditional Quantile UX, and Deterministic MSE operators on 1D Burgers.
- **`train_darcy.sh`**: Trains all four model variants on 2D Darcy Flow using the compact 5x100 architecture with 1024 subsampled sensors.
- **`train_navier_stokes.sh`**: Trains all four model variants on 2D Compressible Navier-Stokes using 1026 branch inputs.
- **`train_biharmonic.sh`**: Trains Jacobian-DeepONet, Vanilla Pinball, and Deterministic MSE operators on 2D Biharmonic FEM data.

Run from the repository root:
```bash
bash examples/training/train_darcy.sh
```

---

## 3. Testing and Benchmark Examples (`examples/testing/`)

These scripts illustrate post-training evaluation and validation:

- **`test_jcb_posthoc.sh`**: Evaluates Jacobian Conformal Bands (JCB) on frozen deterministic operators, demonstrating training-free uncertainty estimation and calibration.
- **`test_uniform_evaluation.sh`**: Evaluates all candidate model variants on held-out test splits with split-conformal coverage equalization.
- **`test_noise_propagation.sh`**: Executes the input Gaussian sensor noise propagation experiment, comparing the analytic Delta method against solver Monte Carlo ground truth.
- **`test_latency_and_montecarlo.sh`**: Profiles CUDA-synchronized inference latency and benchmarks stochastic Monte Carlo ensembles across sample budgets.

Run from the repository root:
```bash
bash examples/testing/test_jcb_posthoc.sh
```
