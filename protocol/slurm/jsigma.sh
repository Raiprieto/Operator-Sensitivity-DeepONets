#!/bin/bash
#SBATCH --job-name=proto_jsigma
#SBATCH --output=logs/protocol/%x_%j.log
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a30mig:1
# Uso: sbatch protocol/slurm/jsigma.sh <benchmark> [--seed N] [--lam L] [--ood NOMBRE] [--smoke]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env_tesis/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u protocol/jacobian_sigma.py "$@"
