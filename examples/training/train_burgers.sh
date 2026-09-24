#!/bin/bash
# ==============================================================================
# Example: Model Training for 1D Burgers Equation
#
# Covers all primary model variants on the same compact architecture:
#   1. Jacobian-DeepONet (Method 1: asymmetric Pinball loss + O(K) latent Gram)
#   2. Vanilla Pinball (controlled ablation: constant width + Pinball loss)
#   3. Spatial Quantile Regression (quantile_ux: conditional branch heads)
#   4. Deterministic Baseline (pure MSE loss, foundation for JCB and MC)
# ==============================================================================
set -euo pipefail

DATA_PATH="data/burgers/burgers_train.h5"
mkdir -p modelos/burgers

# Common architecture: Branch [129, 100, 100], Trunk [2, 100, 100]
LAYER_BRANCH="129 100 100"
LAYER_TRUNK="2 100 100"
BATCH_SIZE=128
ITERS=200000
LR=0.001

echo "1. Training Jacobian-DeepONet on Burgers..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/burgers/burgers_jacobian \
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

echo "2. Training Vanilla Pinball on Burgers..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/burgers/burgers_vanilla \
    --use_cartesian_prod \
    --use_vanilla_pinball \
    --pinball_lambda 4.0 \
    --sigma 0.05 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "3. Training Quantile UX (Conditional Quantile) on Burgers..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/burgers/burgers_qux \
    --use_cartesian_prod \
    --use_quantile_heads ux \
    --pinball_lambda 4.0 \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "4. Training Deterministic DeepONet (MSE) on Burgers..."
python src/model_training/deeponet_training.py \
    --data_path "$DATA_PATH" \
    --output_path modelos/burgers/burgers_mse \
    --use_cartesian_prod \
    --loss_type MSE \
    --layer_sizes_branch $LAYER_BRANCH \
    --layer_sizes_trunk $LAYER_TRUNK \
    --batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --iterations $ITERS

echo "Burgers training suite complete."
