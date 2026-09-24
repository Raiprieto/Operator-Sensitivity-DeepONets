"""
deterministic_deeponet.py

DeepONet determinista: solo predice el centro, no aprende ninguna banda. Es el
modelo BASE de los benchmarks Monte Carlo (results/mc_benchmark.py) y del metodo
Delta, donde la incertidumbre no se entrena sino que se construye despues,
perturbando la entrada y tomando percentiles de las predicciones.

Para que el resto del protocolo (predict, metricas, evaluacion) funcione sin
casos especiales, el forward replica la prediccion en las tres columnas:

    cat([y, y, y])  ->  intervalo de ancho cero

que es exactamente lo que un modelo determinista promete. Sus metricas de UQ se
leen en consecuencia: PICP = 0, MPIW = 0, y el interval score se reduce a
(2/alpha) * |error|, la penalizacion pura por no cubrir. La metrica que importa
para seleccionar su checkpoint es el MSE.
"""

import deepxde as dde
import torch
import torch.nn.functional as F


def create_deterministic_deeponet(is_cartesian=True):
    if not is_cartesian:
        raise NotImplementedError("Solo implementado para el DeepONet cartesiano.")

    class DeterministicDeepONet(dde.nn.DeepONetCartesianProd):
        def forward(self, inputs):
            y = super().forward(inputs)
            return torch.cat([y, y, y], dim=-1)

    return DeterministicDeepONet


def mse_loss_triple(y_true, y_pred):
    """MSE del centro, ignorando las columnas replicadas."""
    n = y_pred.shape[-1] // 3
    return F.mse_loss(y_pred[:, :n], y_true)
