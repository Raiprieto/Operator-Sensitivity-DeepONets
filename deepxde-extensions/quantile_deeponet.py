"""
quantile_deeponet.py

Fair-comparison baseline for Jacobian-DeepONet: a *spatial* deep quantile regression
DeepONet with the SAME backbone (branch + trunk + bias, Cartesian product) and the
SAME composite loss (MSE + lambda * pinball), but whose interval width is learned
directly from data instead of being shaped by the branch Jacobian.

Two conditioning levels are provided, giving a clean ablation ladder together with
the existing models:

    constant-width  (vanilla_pinball_deeponet.py)  width = softplus(w)             [0-dim]
    quantile "x"    (this file, conditioning="x")   width = softplus(h(T(x)))       [depends on x only]
    quantile "ux"   (this file, conditioning="ux")  width = softplus(B_k(u) . T(x)) [depends on u and x]
    Jacobian        (jacobian_deeponet_softplus.py) width = softplus(w) * f(||J_branch||) [physics prior]

Output layout is identical to the other models: cat([y, y_lower, y_upper], dim=-1),
so `pinball_loss_softplus` and all evaluation scripts work unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepxde as dde


def create_spatial_quantile_deeponet(is_cartesian=True):
    if not is_cartesian:
        raise NotImplementedError("SpatialQuantileDeepONet is implemented for the Cartesian-product DeepONet only.")

    BaseClass = dde.nn.DeepONetCartesianProd

    class SpatialQuantileDeepONet(BaseClass):
        def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer,
                     conditioning="ux", **kwargs):
            if conditioning not in ("x", "ux"):
                raise ValueError("conditioning must be 'x' or 'ux'")
            self.conditioning = conditioning
            K = layer_sizes_trunk[-1]

            if conditioning == "ux":
                # split_branch: branch emits 3K latent coefficients (center, lower, upper),
                # trunk basis T(x) in R^K is shared by the three heads.
                lsb = list(layer_sizes_branch)
                if lsb[-1] != 3 * K:
                    lsb[-1] = 3 * K
                super().__init__(
                    layer_sizes_branch=lsb,
                    layer_sizes_trunk=layer_sizes_trunk,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    num_outputs=3,
                    multi_output_strategy="split_branch",
                )
            else:
                # single-output backbone; two linear heads on the trunk basis give a
                # purely spatial (instance-independent) width.
                super().__init__(
                    layer_sizes_branch=layer_sizes_branch,
                    layer_sizes_trunk=layer_sizes_trunk,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                )
                self.head_lower = nn.Linear(K, 1)
                self.head_upper = nn.Linear(K, 1)
                nn.init.zeros_(self.head_lower.weight); nn.init.zeros_(self.head_lower.bias)
                nn.init.zeros_(self.head_upper.weight); nn.init.zeros_(self.head_upper.bias)

        def forward(self, inputs):
            u, x = inputs
            if self._input_transform is not None:
                x = self._input_transform(x)

            T = self.activation_trunk(self.trunk(x))          # [N, K]
            Bf = self.branch(u)                                # [B, K] or [B, 3K]
            K = T.shape[1]

            if self.conditioning == "ux":
                B_c, B_l, B_u = Bf[:, :K], Bf[:, K:2 * K], Bf[:, 2 * K:]
                y = torch.einsum("bi,ni->bn", B_c, T) + self.b[0]
                delta_lower = F.softplus(torch.einsum("bi,ni->bn", B_l, T) + self.b[1])
                delta_upper = F.softplus(torch.einsum("bi,ni->bn", B_u, T) + self.b[2])
            else:
                y = torch.einsum("bi,ni->bn", Bf, T) + self.b[0]
                delta_lower = F.softplus(self.head_lower(T)).T          # [1, N] broadcast over batch
                delta_upper = F.softplus(self.head_upper(T)).T
                delta_lower = delta_lower.expand_as(y)
                delta_upper = delta_upper.expand_as(y)

            y_lower = y - delta_lower
            y_upper = y + delta_upper
            out = torch.cat([y, y_lower, y_upper], dim=-1)
            if self._output_transform is not None:
                out = self._output_transform(inputs, out)
            return out

    return SpatialQuantileDeepONet
