#!/bin/bash
#SBATCH --job-name=proto_gen
#SBATCH --output=logs/protocol/%x_%j.log
#SBATCH --partition=compute-slim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Uso: sbatch protocol/slurm/gen.sh <benchmark> <train|val|test> [--smoke]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env_tesis/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JAX_PLATFORM_NAME=cpu
python -u protocol/generate.py "$@"
