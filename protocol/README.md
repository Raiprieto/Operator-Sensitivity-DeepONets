# Experimental Protocol Engine

A unified, fail-hard protocol for training, validating, and evaluating neural operator uncertainty quantification across all four benchmark equations (Burgers, Biharmonic, Darcy Flow, and Navier-Stokes).

---

## Design Principles

1. **Fail-Hard Execution (Exit Code 3):** The protocol aborts immediately upon any violation:
   - Uncommitted modifications in `src/`, `deepxde-extensions/`, or `protocol/`.
   - Data hash mismatch against signed `MANIFEST.json`.
   - Seed range overlaps between train, validation, and test splits.
   - Non-zero cross-split sample leakage (exact equality or Euclidean distance below $10^{-6}$).
   - Execution outside of Slurm resource allocations.
   - Numerical instability (`NaN` or `Inf` in validation losses).
2. **Strict Test Partition Isolation:** Test sets are never touched during training or hyperparameter tuning. Model checkpoints are selected exclusively via the *interval score* on the validation partition.
3. **Single Source of Truth:** Benchmark configurations in `protocol/configs/<benchmark>.json` govern all parameters: split generators, seed bases, architectures, sigmas, learning rate schedules, and parameter sweeps.

---

## Pipeline Stages

```
gen train ┐
gen val   ├─> verify ─> train (SLURM job array) ─> eval ─> aggregate ─> jcb
gen test  ┘
```

1. **`generate.py`**: Generates individual dataset splits or out-of-distribution evaluation sets (`ood:<name>`) with verified seed offsets.
2. **`verify.py`**: Validates all three splits, verifies zero cross-split sample leakage, performs Kolmogorov-Smirnov distribution checks, and writes the signed `MANIFEST.json`.
3. **`train.py`**: Executes an individual run. Evaluates validation metrics every `val_every` steps and preserves `best.pt` according to the lowest validation interval score. Computes and freezes the reference uncertainty scale (`unc_ref`) over training data.
4. **`evaluate.py`**: Evaluates all trained sweep models on the held-out test split, verifying batch-size invariance and model hashes.
5. **`aggregate.py`**: Compiles final results across all declared random seeds into `summary.md`, `summary.csv`, and `summary.json` reporting mean and standard deviation.
6. **`conformal.py`**: Implements split-conformal rescaling calibrated on an independent half-split to compare prediction interval widths at identical 90% empirical coverage.
7. **`correlations.py`**: Computes Spearman and Pearson correlations against error and physical sensitivity on certified test and OOD splits.

---

## Jacobian Conformal Bands (JCB) Suite

The protocol provides standalone modules for training-free uncertainty quantification on frozen operators:

- **`jacobian_conformal.py`**: Computes Jacobian Conformal Bands (JCB) on a frozen deterministic operator (trained with MSE alone). Compares four candidate band shapes under identical split-conformal calibration:
  - `const_phys`: Standard constant-width conformal band ($h = 1$).
  - `const_norm`: Constant width in normalized output space ($h = Y_{\mathrm{std}}$).
  - `jac`: Jacobian sensitivity geometry ($h = Y_{\mathrm{std}} \cdot \sigma_{\mathrm{norm}} + \beta \, \mathrm{median}$).
  - `jac_log`: Logarithmically compressed Jacobian shape ($h = Y_{\mathrm{std}} \cdot \log(1 + \sigma_{\mathrm{norm}} / \mathrm{ref})$).
- **`jacobian_sigma.py`**: Evaluates whether the latent branch Jacobian of a frozen model already correlates with output error without any uncertainty training.
- **`plot_quartiles.py`**: Generates error-quartile decomposition figures ($Q_1=25\%$, $Q_2=50\%$, $Q_3=75\%$, $Q_4=95\%$) across benchmarks for both trained Jacobian-DeepONet models and training-free JCB bands:
  ```bash
  python protocol/plot_quartiles.py --source trained --out quartiles_trained.png
  python protocol/plot_quartiles.py --source jcb     --out quartiles_jcb.png
  ```
- **`tolerance_conformal.py`**: Computes conformal tolerance regions providing finite-sample coverage guarantees.

---

## Slurm Workflow

The complete execution chain is automated via `protocol/slurm/launch.sh` with `afterok` job dependencies:

```bash
# Fast smoke test (verifies end-to-end pipeline in minutes)
bash protocol/slurm/launch.sh darcy_small --smoke

# Full sweep execution
bash protocol/slurm/launch.sh darcy_small

# Run JCB evaluation on a trained benchmark
sbatch protocol/slurm/jconf.sh darcy_small

# Generate quartile figures
sbatch protocol/slurm/quart.sh
```

---

## Output Layout

```
data/protocol/<benchmark>/
    train.h5, val.h5, test.h5
    MANIFEST.json

runs/protocol/<benchmark>/<model>_lam<lambda>_seed<seed>/
    metadata.json          # Configuration snapshot, git commit hash, and validation metrics
    best.pt                # Checkpoint selected by validation interval score
    scalers.npz            # Z-score normalization statistics
    val_history.csv        # Validation history throughout training
    test_metrics.json      # Final test evaluations
    jacobian_conformal.json # Post-hoc JCB evaluation results

runs/protocol/<benchmark>/
    summary.md             # Aggregated Markdown table across all seeds
    summary.csv            # Tabular results
    summary.json           # Machine-readable output
```
