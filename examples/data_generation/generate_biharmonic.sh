#!/bin/bash
# ==============================================================================
# Example: Data generation for 2D Biharmonic Equation
#
# Exploits machine-precision linearity scaling (f = c * g, u = c * u_base)
# to construct certified, disjoint test sets from an existing base simulation,
# guaranteeing zero sample leakage against the training partition.
# ==============================================================================
set -euo pipefail

mkdir -p data/biharmonic

echo "Generating Biharmonic 2D certified test split (510 samples, seed 20260921)..."
python src/data_generation/biharmonic_scaled_test.py \
    --train_path data/biharmonic/biharmonic_equation.h5 \
    --output_path data/biharmonic/biharmonic_test_clean.h5 \
    --num_samples 510 \
    --seed 20260921

echo "Biharmonic dataset generation complete."
