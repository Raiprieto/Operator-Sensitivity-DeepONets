import deepxde as dde
import torch
import torch.nn.functional as F
import numpy as np


class QuantileDeepONet(dde.nn.DeepONet):
    def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer):
        # num_outputs=3 -> (y, lower, upper)
        super().__init__(
            layer_sizes_branch=layer_sizes_branch,
            layer_sizes_trunk=layer_sizes_trunk,
            activation=activation,
            kernel_initializer=kernel_initializer,
            num_outputs=3,
            multi_output_strategy="split_trunk",  # reuse trunk, split output
        )

    def forward(self, inputs):
        """returns (y, y_lower, y_upper)."""
        out = super().forward(inputs)
        y, y_lower, y_upper = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        return torch.cat([y, y_lower, y_upper], dim=-1)




def quantile_loss(y_true, y_pred, alpha=0.1, lambda_mono=0.1):
    """Función de perdida pinball"""
    y, y_lower, y_upper = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3]

    L_center = F.mse_loss(y, y_true)

    diff_lower = y_true - y_lower
    diff_upper = y_true - y_upper
    L_lower = torch.mean(torch.maximum(alpha / 2 * diff_lower, (alpha / 2 - 1) * diff_lower))
    L_upper = torch.mean(torch.maximum((1 - alpha / 2) * diff_upper, ((1 - alpha / 2) - 1) * diff_upper))

    L_mono = torch.mean(F.relu(y_lower - y) + F.relu(y - y_upper))

    return L_center + L_lower + L_upper + lambda_mono * L_mono