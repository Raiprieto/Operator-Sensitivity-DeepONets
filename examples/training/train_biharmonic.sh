#!/bin/bash
# ==============================================================================
# Example: Model Training for 2D Biharmonic Equation
#
# Trains models on the 4225-sensor Cartesian FEM dataset:
#   1. Jacobian-DeepONet (Method 1: Pinball lambda=4.0 + latent O(K) variance)
#   2. Vanilla Pinball (constant width baseline)
#   3. Deterministic Baseline (pure MSE)
# ==============================================================================
set -euo pipefail

DATA_PATH="data/biharmonic/biharmonic_equation.h5"
mkdir -p modelos/biharmonic

LAYER_BRANCH="4225 100 100 100 100 100"
LAYER_TRUNK="2 100 100 100 100 100"
BATCH_SIZE=128
ITERS=200000
LR=0.001

echo "1. Training Jacobian-DeepONet on Biharmonic 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/biharmonic/biharmonic_jacobian \
    --use_cartesian_prod \
    --use_jacobian \
    --variance_activation softplus \
    --pinball_lambda 4.0 \
    --sigma 0.05 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "2. Training Vanilla Pinball on Biharmonic 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/biharmonic/biharmonic_vanilla \
    --use_cartesian_prod \
    --use_vanilla_pinball \
    --pinball_lambda 4.0 \
    --sigma 0.05 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "3. Training Deterministic DeepONet (MSE) on Biharmonic 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/biharmonic/biharmonic_mse \
    --use_cartesian_prod \
    --loss_type MSE \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "Biharmonic training suite complete."
