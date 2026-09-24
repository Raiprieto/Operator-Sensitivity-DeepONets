# Experimental Protocol Engine

A unified, fail-hard protocol for training, validating, and evaluating neural operator uncertainty quantification across all four benchmark equations (Burgers, Biharmonic, Darcy Flow, and Navier-Stokes).

---

## Design Principles

1. **Fail-Hard Execution (Exit Code 3):** The protocol does not proceed upon warnings. Any violation halts execution immediately with an exit code of 3. This includes:
   - Uncommitted modifications in `src/`, `deepxde-extensions/`, or `protocol/`.
   - Data hash mismatch against `MANIFEST.json`.
   - Seed range overlaps between train, validation, and test splits.
   - Non-zero sample overlap between partitions (exact match or Euclidean distance below $10^{-6}$).
   - Execution outside of Slurm resource allocations.
   - Numerical instability (`NaN` or `Inf` in validation losses).
2. **Strict Test Partition Isolation:** Test sets are never touched during training or hyperparameter tuning. Model checkpoints are selected exclusively via the *interval score* on the validation partition.
3. **Single Source of Truth:** Benchmark configurations in `protocol/configs/<benchmark>.json` govern all parameters: split generators, seed bases, architectures, sigmas, learning rate schedules, and parameter sweeps.

---

## Pipeline Stages

```
gen train ┐
gen val   ├─> verify ─> train (SLURM job array) ─> eval ─> aggregate
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

## Slurm Workflow

The complete execution chain is automated via `protocol/slurm/launch.sh` with `afterok` job dependencies:

```bash
# Fast smoke test (verifies end-to-end pipeline in minutes)
bash protocol/slurm/launch.sh darcy_small --smoke

# Full sweep execution
bash protocol/slurm/launch.sh darcy_small

# Resume execution from a specific stage
bash protocol/slurm/launch.sh darcy_small --from train
```

Supported flags include:
- `--train-gres`: GPU specification for training tasks (default: `gpu:a100:1` or `gpu:a30mig:1`).
- `--eval-gres`: GPU specification for evaluation tasks (default: `gpu:a30mig:1`).
- `--after JOBID`: Chains the initial stage to an already queued Slurm job.

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

runs/protocol/<benchmark>/
    summary.md             # Aggregated Markdown table across all seeds
    summary.csv            # Tabular results
    summary.json           # Machine-readable output
```
