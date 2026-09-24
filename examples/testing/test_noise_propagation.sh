#!/bin/bash
# ==============================================================================
# Example: Input Noise Propagation Experiment
#
# Propagates known Gaussian sensor noise (u0 + eps, eps ~ N(0, sigma^2))
# through the analytic first-order Delta method:
#   Var(x) = T(x) (J_branch Sigma_in J_branch^T) T(x)^T
# Compares against the numerical solver Monte Carlo ground truth and network MC.
# ==============================================================================
set -euo pipefail

mkdir -p results/noise_propagation

echo "Running input noise propagation benchmark on Burgers..."
python benchmarks/input_noise_propagation.py \
    --test data/burgers/burgers_test.h5 \
    --models det=modelos/burgers/burgers_mse \
             qux=modelos/burgers/burgers_qux \
             jac=modelos/burgers/burgers_jacobian \
    --sigmas 0.01 0.05 0.10 \
    --n_inputs 20 \
    --n_mc 1000 \
    --out results/noise_propagation/noise_prop_burgers

echo "Noise propagation experiment complete."
