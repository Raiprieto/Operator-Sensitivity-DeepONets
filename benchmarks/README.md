# Evaluation Benchmarks Suite

This directory contains standalone benchmark utilities and diagnostic pipelines for uncertainty quantification, sensitivity correlation, latency profiling, and input noise propagation.

---

## Benchmark Scripts

### 1. Jacobian Conformal Bands (JCB) Post-Hoc Benchmark (`posthoc_jacobian.py`)
Implements the training-free Jacobian Conformal Bands (JCB) framework on already-trained deterministic DeepONet operators:
- **Zero Retraining:** Extracts the latent branch Jacobian from a plain `DeepONetCartesianProd` (trained exclusively with MSE loss) via $K$ reverse-mode vector-Jacobian products.
- **Sensitivity Shaping:** Computes the exact pointwise sensitivity template $s(x) = \sqrt{\mathrm{diag}(T(x) G T(x)^T)}$ and normalizes it over the calibration set.
- **Comparative Shapes:** Evaluates multiple non-parametric band geometries under identical split-conformal calibration:
  - `const`: Constant width (standard split-conformal prediction baseline).
  - `posthoc`: Raw linear Jacobian sensitivity shape.
  - `posthoc_log1p`: Logarithmically compressed Jacobian shape $\log(1 + s / \bar{s})$.
  - `posthoc_persample`: Per-sample localized shape ablation.
  - `mc_perturb_M64`: Empirical perturbation Monte Carlo baseline through the deterministic backbone.

```bash
python benchmarks/posthoc_jacobian.py \
    --test data/burgers_test_500.h5 \
    --det modelos/burgers_det \
    --bands qux=modelos/burgers_qux jac=modelos/burgers_jac \
    --out results/posthoc_burgers
```

### 2. Input Noise Propagation (`input_noise_propagation.py`)
Evaluates the task only a Jacobian-based operator can perform in one forward pass: propagating known input Gaussian sensor noise to the output field via the first-order Delta method:
$$\mathrm{Var}(x) = T(x) (J_{branch} \, \Sigma_{in} \, J_{branch}^T) T(x)^T$$
Compares the analytic Delta method against:
- High-fidelity Monte Carlo simulations directly through the numerical solver (ground truth).
- Empirical Monte Carlo forward passes through the neural operator.

```bash
python benchmarks/input_noise_propagation.py \
    --test data/burgers_test_500.h5 \
    --models det=modelos/burgers_det jac=modelos/burgers_jac \
    --out results/noise_prop_burgers
```

### 3. Uniform Model Evaluation and Conformal Calibration (`fair_eval.py`)
Computes standardized metrics in physical units across candidate models on test sets:
- Mean Squared Error (MSE) and Relative L2 Error
- Empirical Prediction Interval Coverage Probability (PICP at 90% nominal level)
- Mean Prediction Interval Width (MPIW)
- Spearman rank correlation against error (rho_Err) and true physical sensitivity (rho_Sens)
- Decile stratification analysis across sample difficulty (D1 through D10)
- Split-conformal rescaling for width comparisons at equalized coverage

### 4. Stochastic Monte Carlo Benchmark (`mc_benchmark.py`)
Executes Monte Carlo uncertainty quantification over baseline deterministic models across sample sizes (N_MC in {100, 500, 1000}) to quantify empirical coverage, interval widths, and computational latency:
```bash
python benchmarks/mc_benchmark.py \
    --data_path data/darcy_test.h5 \
    --model_dir modelos/darcy_mse \
    --n_mc 500 \
    --alpha 0.10
```

### 5. Latency Profiling (`benchmark_jacobian_latency.py`)
Profiles inference latency with CUDA synchronization and warm-up passes to establish wall-clock speedup metrics comparing single-pass Jacobian extraction against stochastic ensembles.

### 6. Sensitivity Floor Analysis (`rho_sens_floor.py`)
Computes the theoretical floor of rho_Sens induced by spatial output normalizations (Z-score scalers) to prevent misleading baseline comparisons.
