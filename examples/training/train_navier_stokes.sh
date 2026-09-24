#!/bin/bash
# ==============================================================================
# Example: Model Training for 2D Compressible Navier-Stokes (Flow Map)
#
# Covers all primary model variants on the compact 5x100 architecture:
#   1. Jacobian-DeepONet (Method 1: Pinball lambda=4.0 + latent O(K) variance)
#   2. Vanilla Pinball (constant width baseline)
#   3. Spatial Quantile Regression (quantile_ux: conditional branch head)
#   4. Deterministic Baseline (pure MSE, foundation for JCB and MC)
# ==============================================================================
set -euo pipefail

DATA_PATH="data/ns2d/ns2d_train.h5"
mkdir -p modelos/ns2d

# 1026 inputs: 1024 density sensors (stride 2) + Gamma + mu
LAYER_BRANCH="1026 100 100 100 100 100"
LAYER_TRUNK="2 100 100 100 100 100"
BATCH_SIZE=128
ITERS=200000
LR=0.001

echo "1. Training Jacobian-DeepONet on Navier-Stokes 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/ns2d/ns2d_jacobian \
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

echo "2. Training Vanilla Pinball on Navier-Stokes 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/ns2d/ns2d_vanilla \
    --use_cartesian_prod \
    --use_vanilla_pinball \
    --pinball_lambda 4.0 \
    --sigma 0.05 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "3. Training Quantile UX on Navier-Stokes 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/ns2d/ns2d_qux \
    --use_cartesian_prod \
    --use_quantile_heads ux \
    --pinball_lambda 4.0 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "4. Training Deterministic DeepONet (MSE) on Navier-Stokes 2D..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/ns2d/ns2d_mse \
    --use_cartesian_prod \
    --loss_type MSE \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "Navier-Stokes training suite complete."
