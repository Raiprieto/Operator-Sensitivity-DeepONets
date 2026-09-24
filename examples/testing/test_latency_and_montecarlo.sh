#!/bin/bash
# ==============================================================================
# Example: Latency Profiling and Monte Carlo Benchmarks
#
# Profiles wall-clock execution time and speedup of analytic Jacobian extraction
# against stochastic Monte Carlo forward ensembles (N_MC = 100, 500, 1000).
# ==============================================================================
set -euo pipefail

mkdir -p results/latency results/mc

echo "1. Profiling inference latency on Darcy Flow (GPU with CUDA synchronization)..."
python benchmarks/benchmark_jacobian_latency.py \
    --equation_name darcy \
    --data_path data/darcy/darcy_test.h5 \
    --model_dir_baseline modelos/darcy/darcy_mse \
    --model_dir_jacobian modelos/darcy/darcy_jacobian \
    --model_dir_vanilla modelos/darcy/darcy_vanilla \
    --num_samples 100 \
    --device cuda

echo "2. Running Monte Carlo uncertainty sampling benchmark on Darcy Flow..."
python benchmarks/mc_benchmark.py \
    --data_path data/darcy/darcy_test.h5 \
    --model_dir modelos/darcy/darcy_mse \
    --n_mc 500 \
    --alpha 0.10 \
    --device cuda

echo "Latency and Monte Carlo benchmarks complete."
