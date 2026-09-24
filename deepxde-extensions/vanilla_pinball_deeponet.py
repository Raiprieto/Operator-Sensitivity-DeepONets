"""
vanilla_pinball_deeponet.py

Ablacion de la Jacobian-DeepONet: arquitectura IDENTICA (una branch, un trunk,
w_lower, w_upper), pero sin computar el Jacobiano. Las bandas de incertidumbre
son constantes (no tienen estructura espacial), y su ancho es controlado
unicamente por softplus(w) * sigma.

Sirve para aislar la contribucion del Jacobiano: si la version con Jacobiano
tiene mejor correlacion espacial y cobertura comparable, el Jacobiano aporta
informacion fisica que no se puede reemplazar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import deepxde as dde


def create_vanilla_pinball_deeponet(is_cartesian=False):
    """Crea una DeepONet con bandas de incertidumbre constantes (sin Jacobiano).

    Arquitectura identica a JacobianDeepONetSoftplus:
      - Una Branch Network
      - Una Trunk Network
      - Dos escalares entrenables (w_lower, w_upper)
      - sigma como hiperparametro

    La diferencia: en vez de derivar la forma de la incertidumbre del Jacobiano
    (J_branch), usa bandas uniformes. El Pinball Loss entrena w para encontrar
    el ancho global optimo.

    Args:
        is_cartesian: Si True, usa DeepONetCartesianProd como base.

    Returns:
        Clase VanillaPinballDeepONet.
    """
    BaseClass = dde.nn.DeepONetCartesianProd if is_cartesian else dde.nn.DeepONet

    class VanillaPinballDeepONet(BaseClass):
        def __init__(self, layer_sizes_branch, layer_sizes_trunk,
                     activation, kernel_initializer, sigma=0.05, **kwargs):
            """Inicializa la red con la misma estructura que JacobianDeepONetSoftplus.

            Args:
                layer_sizes_branch: Dimensiones de la Branch Network [input, h1, ..., K].
                layer_sizes_trunk: Dimensiones de la Trunk Network [input, h1, ..., K].
                activation: Funcion de activacion (str).
                kernel_initializer: Inicializador de pesos (str).
                sigma: Factor de escala de incertidumbre (mismo rol que en Jacobian-DeepONet).
            """
            super().__init__(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            self.sigma = sigma

            # Mismos pesos asimetricos entrenables que JacobianDeepONetSoftplus
            # Inicializados en 0 para que softplus(0) ~ 0.69
            self.w_lower = nn.Parameter(torch.zeros(1))
            self.w_upper = nn.Parameter(torch.zeros(1))

        def forward(self, inputs):
            u, x = inputs

            # Prediccion central (identica a la Jacobian-DeepONet)
            y = super().forward((u, x))

            # SIN JACOBIANO: bandas de ancho constante.
            # La forma espacial de la incertidumbre es uniforme (1.0 en todos los puntos).
            # Solo softplus(w) * sigma controla el ancho global.
            delta_lower = F.softplus(self.w_lower) * self.sigma
            delta_upper = F.softplus(self.w_upper) * self.sigma

            y_lower = y - delta_lower
            y_upper = y + delta_upper

            return torch.cat([y, y_lower, y_upper], dim=-1)

    return VanillaPinballDeepONet
