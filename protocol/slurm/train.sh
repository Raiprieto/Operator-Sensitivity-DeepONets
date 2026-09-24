#!/bin/bash
#SBATCH --job-name=proto_train
#SBATCH --output=logs/protocol/%x_%A_%a.log
#SBATCH --partition=compute-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
# Uso: sbatch --array=0-(N-1) protocol/slurm/train.sh <benchmark> [--smoke]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source env_tesis/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -u protocol/train.py "$1" --index "$SLURM_ARRAY_TASK_ID" ${2:-}
