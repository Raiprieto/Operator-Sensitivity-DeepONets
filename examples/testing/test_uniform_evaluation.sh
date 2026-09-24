#!/bin/bash
# ==============================================================================
# Example: Uniform Model Evaluation and Split-Conformal Rescaling
#
# Evaluates all candidate model variants on physical units:
#   MSE, PICP, MPIW, Gaussian proxy-NLL, rho_Err, rho_Sens, and decile coverage.
# ==============================================================================
set -euo pipefail

mkdir -p results/fair

echo "Running uniform evaluation on Burgers test split..."
python benchmarks/fair_eval.py \
    --test data/burgers/burgers_test.h5 \
    --models det=modelos/burgers/burgers_mse \
             van=modelos/burgers/burgers_vanilla \
             qux=modelos/burgers/burgers_qux \
             jac=modelos/burgers/burgers_jacobian \
    --out results/fair/burgers_fair_eval

echo "Uniform evaluation complete. Results written to results/fair/burgers_fair_eval.md and .json"
