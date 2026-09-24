# Core Source Code: Data Generation and Model Training

This directory contains the physical PDE data generators and the procedural model training pipeline.

---

## Directory Organization

```
src/
├── data_generation/      # Numerical PDE solvers and certified dataset generators
└── model_training/       # Procedural model construction, optimization, and serialization
```

---

## Data Generation (`src/data_generation/`)

All data generators enforce deterministic seed tracking and prevent training/test contamination:

### 1. 1D Burgers Equation (`burgers_pdebench.py`)
Simulates the non-linear advection-diffusion equation:
$$\partial_t u + u \partial_x u = \nu \partial_{xx} u$$
- **Solver Scheme:** Second-order Finite Volume Method (FVM) with Van Leer flux limiter.
- **Sensitivity Reference:** Central finite differences over kinematic viscosity ($\Delta \nu = 10^{-4}$), avoiding non-differentiable limiter artifacts (`abs`, `sign`) in reverse-mode autodiff.

### 2. 2D Darcy Flow (`darcy_pdebench.py`)
Simulates pressure fields in porous media:
$$-\nabla \cdot (K(x,y) \nabla u(x,y)) = 1, \quad (x,y) \in [0,1]^2$$
- **Ellipticity:** Guaranteed by defining permeability fields as $K(x,y) = \exp(a(x,y))$, where $a(x,y)$ is a Gaussian Random Field (GRF).
- **Out-of-Distribution Control:** Adjustable via `--grf_scale` (standard value: 0.5; out-of-domain perturbations: > 0.5).
- **Implicit Sensitivity:** Ground-truth Jacobian is computed using JAX implicit differentiation on conjugate gradient solves. Supports `--jacobian_mode sum` to extract row sensitivities in a single forward JVP pass.

### 3. 2D Compressible Navier-Stokes (`navier_stokes_state_pdebench.py`)
Evaluates the final-state Flow Map ($\mathcal{G}: \rho_0 \mapsto \rho_{T=0.5}$) for compressible gas dynamics:
- **Solver:** Explicit predictor-corrector MacCormack scheme with entropy-stable pressure clipping.
- **Branch Dimensions:** 4098 inputs (4096 initial density sensors on a 64x64 grid, adiabatic coefficient $\Gamma$, and dynamic viscosity $\mu$).
- **Parametric Sensitivity:** Central finite differences with respect to $\Gamma$.

### 4. 2D Biharmonic Equation (`biharmonic_scaled_test.py`)
Generates certified, non-overlapping test partitions for the fourth-order linear plate equation ($\Delta^2 u = f$) via exact machine-precision linearity scaling ($f = c \cdot g$).

### 5. Leakage Verification (`check_leakage.py`)
Audits dataset files to verify zero intersection between train, validation, and test splits (exact equality and Euclidean distances below $\epsilon = 10^{-6}$).

---

## Model Training (`src/model_training/`)

### `deeponet_training.py`
Centralized procedural training engine:
- **Unified Architecture Factory:** Spawns `JacobianDeepONet`, `VanillaPinballDeepONet`, `QuantileDeepONet`, or deterministic DeepONets based on CLI arguments.
- **Z-Score Normalization:** Computes and saves input and target statistics (`<output>_scalers.npz`) fitted exclusively on the training partition.
- **Metadata Serialization:** Exports complete execution arguments to `<output>_args.json` for reproducibility and test evaluation.
- **Optimization:** Supports Adam optimization with step learning rate decay schedules, gradient clipping, periodic checkpoints (`--ckpt_every`), and optional L-BFGS refinement.
