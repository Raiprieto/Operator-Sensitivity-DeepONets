#!/bin/bash
# ==============================================================================
# Example: Data generation for 2D Compressible Navier-Stokes (Flow Map)
#
# Simulates the final-state flow map G: rho_0 -> rho_{T=0.5} on a 64x64 grid.
# Inputs: 4096 initial density sensors + adiabatic coefficient Gamma + viscosity mu.
# Physical sensitivity: d rho / d Gamma computed via central finite differences.
# ==============================================================================
set -euo pipefail

mkdir -p data/ns2d

echo "Generating Navier-Stokes 2D training split (500 samples, seed 3000000)..."
python src/data_generation/navier_stokes_state_pdebench.py \
    --num_samples 500 \
    --nx 64 \
    --T 0.5 \
    --output_path data/ns2d/ns2d_train.h5 \
    --batch_size 20 \
    --seed_offset 3000000 \
    --amp_scale 1.0 \
    --skip_jacobian

echo "Generating Navier-Stokes 2D test split with physical sensitivity (50 samples, seed 1000000)..."
python src/data_generation/navier_stokes_state_pdebench.py \
    --num_samples 50 \
    --nx 64 \
    --T 0.5 \
    --output_path data/ns2d/ns2d_test.h5 \
    --batch_size 10 \
    --seed_offset 1000000 \
    --amp_scale 1.0

echo "Generating Navier-Stokes 2D OOD test split with amplified initial density (50 samples, seed 1100000)..."
python src/data_generation/navier_stokes_state_pdebench.py \
    --num_samples 50 \
    --nx 64 \
    --T 0.5 \
    --output_path data/ns2d/ns2d_ood_amp15.h5 \
    --batch_size 10 \
    --seed_offset 1100000 \
    --amp_scale 1.5

echo "Navier-Stokes dataset generation complete."
