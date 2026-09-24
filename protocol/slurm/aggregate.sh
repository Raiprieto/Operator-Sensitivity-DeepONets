#!/bin/bash
#SBATCH --job-name=proto_aggregate
#SBATCH --output=logs/protocol/%x_%j.log
#SBATCH --partition=compute-slim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# Uso: sbatch protocol/slurm/aggregate.sh <benchmark> [--smoke]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env_tesis/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u protocol/aggregate.py "$@"
