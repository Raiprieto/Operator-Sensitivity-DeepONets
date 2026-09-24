#!/bin/bash
# ==============================================================================
# Example: Data generation for 1D Burgers Equation
#
# Generates certified, non-overlapping train and test splits.
# Train split: fast generation omitting reference Jacobian (not used in training).
# Test split: extracts ground-truth physical sensitivities (d u / d nu) via FDM.
# ==============================================================================
set -euo pipefail

mkdir -p data/burgers

echo "Generating Burgers 1D training split (1000 samples, seed 3000000)..."
python src/data_generation/burgers_pdebench.py \
    --num_samples 1000 \
    --output_path data/burgers/burgers_train.h5 \
    --batch_size 20 \
    --seed_offset 3000000 \
    --skip_jacobian

echo "Generating Burgers 1D test split with physical sensitivity (100 samples, seed 1000000)..."
python src/data_generation/burgers_pdebench.py \
    --num_samples 100 \
    --output_path data/burgers/burgers_test.h5 \
    --batch_size 10 \
    --seed_offset 1000000

echo "Burgers dataset generation complete."
