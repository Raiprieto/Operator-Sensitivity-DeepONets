#!/bin/bash
#SBATCH --job-name=proto_eval
#SBATCH --output=logs/protocol/%x_%j.log
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a30mig:1
# Uso: sbatch protocol/slurm/eval.sh <benchmark> [--smoke]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env_tesis/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u protocol/evaluate.py "$@"
