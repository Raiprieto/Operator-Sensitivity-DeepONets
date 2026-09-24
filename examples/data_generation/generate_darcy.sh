#!/bin/bash
# ==============================================================================
# Example: Data generation for 2D Darcy Flow
#
# Solves -div(K grad u) = 1 on [0,1]^2 on a 64x64 grid.
# Permeability is generated as K = exp(a), where a is a Gaussian Random Field.
# Demonstrates in-distribution and out-of-distribution (OOD) generation.
# ==============================================================================
set -euo pipefail

mkdir -p data/darcy

echo "Generating Darcy 2D training split (1000 samples, seed 3000000)..."
python src/data_generation/darcy_pdebench.py \
    --num_samples 1000 \
    --N 64 \
    --output_path data/darcy/darcy_train.h5 \
    --batch_size 10 \
    --seed_offset 3000000 \
    --grf_scale 0.5 \
    --skip_jacobian

echo "Generating Darcy 2D test split with row-sum sensitivities (100 samples, seed 1000000)..."
python src/data_generation/darcy_pdebench.py \
    --num_samples 100 \
    --N 64 \
    --output_path data/darcy/darcy_test.h5 \
    --batch_size 10 \
    --seed_offset 1000000 \
    --jacobian_mode sum \
    --grf_scale 0.5

echo "Generating Darcy 2D OOD test split with enhanced GRF scale (50 samples, seed 1100000)..."
python src/data_generation/darcy_pdebench.py \
    --num_samples 50 \
    --N 64 \
    --output_path data/darcy/darcy_ood_grf075.h5 \
    --batch_size 10 \
    --seed_offset 1100000 \
    --jacobian_mode sum \
    --grf_scale 0.75

echo "Darcy dataset generation complete."
