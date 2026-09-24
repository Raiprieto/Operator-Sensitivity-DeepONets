#!/bin/bash
# ==============================================================================
# Example: Evaluation of Jacobian Conformal Bands (JCB) (Method 2)
#
# Evaluates training-free uncertainty quantification on a frozen deterministic
# operator (trained with pure MSE loss) across Burgers, Darcy, and Navier-Stokes:
#   1. protocol/jacobian_conformal.py (evaluates const_phys, const_norm, jac, jac_log)
#   2. benchmarks/posthoc_jacobian.py (generates detailed post-hoc metrics and ablations)
# ==============================================================================
set -euo pipefail

mkdir -p results/jcb

echo "1. Evaluating JCB via the protocol engine on Darcy Flow..."
python protocol/jacobian_conformal.py darcy_small --seed 0 --beta 0.05

echo "2. Evaluating JCB via the standalone post-hoc benchmark on Burgers..."
python benchmarks/posthoc_jacobian.py \
    --test data/burgers/burgers_test.h5 \
    --det modelos/burgers/burgers_mse \
    --bands qux=modelos/burgers/burgers_qux jac=modelos/burgers/burgers_jacobian \
    --out results/jcb/posthoc_burgers

echo "JCB evaluation complete."
