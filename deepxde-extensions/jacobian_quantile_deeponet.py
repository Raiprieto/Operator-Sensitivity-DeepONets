"""
jacobian_quantile_deeponet.py

Hibrido: el jacobiano del branch fija la FORMA de la incertidumbre (prior fisico)
y unas cabezas con capacidad, al estilo de quantile_deeponet.py (conditioning
"ux"), aprenden una CORRECCION multiplicativa sobre esa forma.

    Jacobian (actual)  delta = softplus(w) * shape(u,x) * sigma          [2 escalares]
    quantile ux        delta = softplus(B_l(u) . T(x) + b)               [~2*K*K params]
    este archivo       delta = softplus(w) * shape(u,x) * g(u,x) * sigma [ambos]

                       g = exp(clamp(h, -3, 3))

Dos niveles de correccion (conditioning):
    "ux"  h(u,x) = (W B(u)) . T(x)   ~2*K*K params: corrige por muestra y por punto
    "x"   h(x)   = w . T(x)          ~2*K params:   corrige solo un perfil espacial

La variante "x" es la interesante conceptualmente: el prior fisico ya aporta toda
la dependencia de la muestra (via ||J(u)||), asi que lo aprendido se limita a
corregir un sesgo espacial sistematico de esa forma, con dos ordenes de magnitud
menos de parametros.

Con centered=True la correccion se centra: g = exp(h - ref), donde ref es la media
de h (del batch durante el entrenamiento, y una media movil congelada en
evaluacion, para que la prediccion no dependa del batch). Asi `g` SOLO puede
redistribuir el ancho, nunca escalarlo de forma global: la escala global queda
exclusivamente en softplus(w). Sin centrar, ambas parametrizan lo mismo y el
optimizador satura el clamp con una constante, que es una reparametrizacion de w.

`W` se inicializa en cero, asi que g = 1 y el modelo arranca siendo EXACTAMENTE
el Jacobian-DeepONet. Las cabezas solo pueden aprender correcciones sobre el
prior, no reemplazarlo, y `h` es interpretable: es la correccion en escala
logaritmica que la red considera necesaria sobre lo que dice la fisica.
Su magnitud (se registra en `self.last_gate_log_abs`) responde de forma
cuantitativa cuanto de la estructura de la incertidumbre ya estaba en el prior.

Cabezas asimetricas (una para el borde inferior y otra para el superior), como
los w_lower / w_upper que reemplazan. Salida identica a los demas modelos:
cat([y, y_lower, y_upper], dim=-1), asi que la perdida y la evaluacion no cambian.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus

GATE_CLAMP = 3.0      # la correccion multiplicativa queda en [e^-3, e^3]


def create_jacobian_quantile_deeponet(is_cartesian=True):
    if not is_cartesian:
        raise NotImplementedError("Solo implementado para el DeepONet cartesiano.")

    BaseJac = create_jacobian_deeponet_softplus(is_cartesian=True)

    class JacobianQuantileDeepONet(BaseJac):
        def __init__(self, layer_sizes_branch, layer_sizes_trunk, activation,
                     kernel_initializer, jacobian_type="parameter", sigma=0.05,
                     conditioning="ux", centered=False):
            if conditioning not in ("x", "ux"):
                raise ValueError("conditioning debe ser 'x' o 'ux'")
            super().__init__(layer_sizes_branch=layer_sizes_branch,
                             layer_sizes_trunk=layer_sizes_trunk,
                             activation=activation,
                             kernel_initializer=kernel_initializer,
                             jacobian_type=jacobian_type,
                             sigma=sigma)
            K = layer_sizes_trunk[-1]
            self.conditioning = conditioning
            # Cero => g = 1 => identico al Jacobian en el paso 0.
            out = K if conditioning == "ux" else 1
            self.head_lower = nn.Linear(K, out, bias=False)
            self.head_upper = nn.Linear(K, out, bias=False)
            nn.init.zeros_(self.head_lower.weight)
            nn.init.zeros_(self.head_upper.weight)
            self.centered = centered
            if centered:
                # Media movil de h, congelada en evaluacion (analogo a unc_ref).
                self.register_buffer("gate_log_mean", torch.zeros(2))
                self.register_buffer("gate_log_init", torch.zeros(1, dtype=torch.bool))
            self.last_gate_log_abs = 0.0

        def _gates(self, u, x):
            """Correccion multiplicativa por muestra y por punto: (g_lower, g_upper)."""
            B = self.branch(u)                                    # [batch, K]
            T = self.trunk(x)
            if hasattr(self, "activation_trunk"):
                T = self.activation_trunk(T)                      # [N_nodes, K]
            gates = []
            for i, head in enumerate((self.head_lower, self.head_upper)):
                if self.conditioning == "ux":
                    h = torch.einsum("bk,nk->bn", head(B), T)       # [batch, N_nodes]
                else:
                    h = head(T).T.expand(B.shape[0], -1)            # [1, N] -> [batch, N]
                if self.centered:
                    if self.training:
                        # SIN detach: asi la derivada en la direccion uniforme es
                        # exactamente cero y el optimizador no puede hacer derivar h
                        # (con detach, el valor queda centrado pero el gradiente no,
                        # h deriva sin efecto en la perdida y termina saturando el clamp).
                        ref = h.mean()
                        with torch.no_grad():
                            m = ref.detach().reshape(())
                            if not bool(self.gate_log_init):
                                self.gate_log_mean[i] = m
                                if i == 1:
                                    self.gate_log_init.fill_(True)
                            else:
                                self.gate_log_mean[i] = 0.99 * self.gate_log_mean[i] + 0.01 * m
                    else:
                        ref = self.gate_log_mean[i]
                    h = h - ref
                gates.append(torch.exp(h.clamp(-GATE_CLAMP, GATE_CLAMP)))
            self.last_gate_log_abs = float(
                torch.log(torch.stack(gates)).abs().mean().detach())
            return gates

        def forward(self, inputs):
            u, x = inputs
            if not u.requires_grad:
                u.requires_grad_(True)
            if not x.requires_grad:
                x.requires_grad_(True)

            y = super(BaseJac, self).forward((u, x))              # centro (backbone DeepONet)

            if getattr(self, "_skip_uncertainty", False):
                self._last_jac_norm = 0.0
                return torch.cat([y, y, y], dim=-1)

            uncertainty = self._compute_uncertainty(y, u, x, True)

            # Misma normalizacion que el Jacobian: media del batch, o referencia fija
            unc_ref = getattr(self, "unc_ref", None)
            if unc_ref is None:
                unc_scale = uncertainty.mean().detach() + 1e-8
            else:
                unc_scale = torch.as_tensor(unc_ref, dtype=uncertainty.dtype,
                                            device=uncertainty.device) + 1e-8
            unc_shape = torch.log1p(uncertainty / unc_scale)
            self._last_jac_norm = float(unc_scale / (self.sigma + 1e-8))

            g_lower, g_upper = self._gates(u, x)
            delta_lower = F.softplus(self.w_lower) * unc_shape * g_lower * self.sigma
            delta_upper = F.softplus(self.w_upper) * unc_shape * g_upper * self.sigma
            return torch.cat([y, y - delta_lower, y + delta_upper], dim=-1)

    return JacobianQuantileDeepONet
