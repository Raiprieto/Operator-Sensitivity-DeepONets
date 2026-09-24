import torch
import torch.nn as nn
import torch.nn.functional as F
import deepxde as dde

def create_jacobian_deeponet_softplus(is_cartesian=False):
    BaseClass = dde.nn.DeepONetCartesianProd if is_cartesian else dde.nn.DeepONet
    
    class JacobianDeepONetSoftplus(BaseClass):
        def __init__(self, layer_sizes_branch, layer_sizes_trunk, 
                     activation, kernel_initializer, jacobian_type="parameter", sigma=0.05,
                     scale_mode="batch", ema_momentum=0.99):
            super().__init__(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            self.sigma = sigma
            self.jacobian_type = jacobian_type
            self.scale_mode = scale_mode
            self.ema_momentum = ema_momentum
            if scale_mode == "ema":
                self.register_buffer("unc_scale_ema", torch.zeros(1))
                self.register_buffer("unc_scale_init", torch.zeros(1, dtype=torch.bool))
            
            # Pesos asimétricos entrenables
            # Inicializados en 0 para que softplus(0) ≈ 0.69
            self.w_lower = nn.Parameter(torch.zeros(1))
            self.w_upper = nn.Parameter(torch.zeros(1))

        def _compute_uncertainty(self, y, u, x, is_cartesian):
            if is_cartesian and self.jacobian_type == "parameter":
                B = self.branch(u)   # [batch, K]
                
                T_raw = self.trunk(x)    # [N_nodes, K]
                T = self.activation_trunk(T_raw) if hasattr(self, "activation_trunk") else T_raw
                
                K = B.shape[-1]

                # Jacobiano del branch: Vectorizado con jacrev
                try:
                    from torch.func import vmap, jacrev
                    
                    def _compute_branch_single(u_s):
                        return self.branch(u_s.unsqueeze(0)).squeeze(0)
                    
                    J_branch = vmap(jacrev(_compute_branch_single))(u)
                except ImportError:
                    J_branch = []
                    for k in range(K):
                        e_k = torch.zeros_like(B)
                        e_k[:, k] = 1.0
                        grad_k = torch.autograd.grad(
                            B, u,
                            grad_outputs=e_k,
                            create_graph=True,
                            retain_graph=True
                        )[0]
                        J_branch.append(grad_k)  
                    J_branch = torch.stack(J_branch, dim=1)  

                # Varianza pointwise exacta mediante Matriz de Gram Latente
                # Acople activado: NO detachamos J_branch, el gradiente fluye al Hessiano
                
                # PROTECCIÓN HESSIANO PROFUNDO: Clamp duro a los valores extremos del Jacobiano.
                # Si una rama de 10 capas intenta explotar hacia infinito, se trunca en 50 
                # y su derivada (Hessiano) se vuelve exactamente 0, cortando la explosión.
                J_branch = torch.clamp(J_branch, min=-50.0, max=50.0)

                # Normalizar J_branch por su magnitud maxima antes de G = J @ J^T
                # para prevenir overflow de float32 (valores > 1e19 desbordan al cuadrado)
                # DETACH: Evita que la cadena del Hessiano fluya por el denominador hacia toda la rama
                j_max = J_branch.abs().max().clamp(min=1.0).detach()
                J_norm = J_branch / j_max

                # ACOPLE TOTAL RECUPERADO: Quitamos el .detach() de G.
                # El gradiente del Pinball DEBE fluir por la Branch para que la red 
                # aprenda a moldear geométricamente el Jacobiano a las zonas de error.
                G = torch.bmm(J_norm, J_norm.transpose(1, 2))
                TG = torch.einsum('nk,bkm->bnm', T, G)
                variance_norm = torch.einsum('bnm,nm->bn', TG, T)
                
                # PREVENCIÓN CRÍTICA DE NaNs:
                # La varianza numérica puede ser ligeramente negativa (ej. -1e-7) por errores de float32 en einsum.
                # Si se multiplica por un j_max grande (ej. 1000), se convierte en -0.1.
                # Al hacer sqrt(-0.1) da NaN. Solución: ReLU/clamp en la varianza normalizada.
                variance_norm = F.relu(variance_norm)
                
                # Extraemos j_max fuera de la raíz para evitar j_max**2 que causa Infs
                uncertainty = j_max * self.sigma * torch.sqrt(variance_norm + 1e-8)
                return uncertainty

            elif self.jacobian_type == "parameter":
                num_n = y.shape[-1]
                grad = torch.autograd.grad(
                    y, u,
                    grad_outputs=torch.ones_like(y) / num_n,
                    create_graph=True,
                    retain_graph=True
                )[0]
                variance = torch.sum(grad**2, dim=-1, keepdim=True) * (self.sigma**2)
            else:
                variance = torch.zeros_like(y)

            return torch.sqrt(F.relu(variance) + 1e-8)  # [batch, N_nodes]

        def forward(self, inputs):
            u, x = inputs

            if not u.requires_grad:
                u.requires_grad_(True)
            if not x.requires_grad:
                x.requires_grad_(True)

            y = super().forward((u, x))

            # Durante warmup del lambda adaptativo, omitir el Jacobiano (costoso)
            if getattr(self, '_skip_uncertainty', False):
                self._last_jac_norm = 0.0
                return torch.cat([y, y, y], dim=-1)

            uncertainty = self._compute_uncertainty(y, u, x, is_cartesian)

            # Normalizacion de la forma de la incertidumbre:
            # - Si scale_mode == "ema": normaliza por media movil exponencial congelada en eval.
            # - Si unc_ref esta definido: referencia fija determinista por muestra (estandar protocol).
            # - Por defecto: media del batch actual.
            batch_scale = uncertainty.mean().detach() + 1e-8
            if self.scale_mode == "ema":
                if self.training:
                    if not bool(self.unc_scale_init):
                        self.unc_scale_ema.copy_(batch_scale.reshape(1))
                        self.unc_scale_init.fill_(True)
                    else:
                        self.unc_scale_ema.mul_(self.ema_momentum).add_((1 - self.ema_momentum) * batch_scale.reshape(1))
                unc_scale = self.unc_scale_ema.reshape(()) if bool(self.unc_scale_init) else batch_scale
            else:
                unc_ref = getattr(self, "unc_ref", None)
                if unc_ref is not None:
                    unc_scale = torch.as_tensor(unc_ref, dtype=uncertainty.dtype,
                                                device=uncertainty.device) + 1e-8
                else:
                    unc_scale = batch_scale
            unc_shape = uncertainty / unc_scale

            # Amortiguación Logarítmica: Evita la explosión del Hessiano y mantiene el acople
            unc_shape = torch.log1p(unc_shape)

            # Norma pura del Jacobiano (sin sigma) para métricas (opcional)
            self._last_jac_norm = (unc_scale / (self.sigma + 1e-8)).item()

            delta_lower = F.softplus(self.w_lower) * unc_shape * self.sigma
            delta_upper = F.softplus(self.w_upper) * unc_shape * self.sigma

            # Acople Total: Pinball entrena w, unc_shape (Hessiano estabilizado por log) e y central
            y_lower = y - delta_lower
            y_upper = y + delta_upper

            return torch.cat([y, y_lower, y_upper], dim=-1)

    return JacobianDeepONetSoftplus


def pinball_loss_softplus(y_true, y_pred, pinball_lambda=1.0):
    """Pérdida Pinball con lambda configurable.
    
    Args:
        pinball_lambda: Multiplicador de la pérdida Pinball respecto al MSE central.
                        Valores altos (ej. 10) fuerzan bandas más anchas.
    """
    alpha = 0.1  # 90% de cobertura objetivo

    num_nodes = y_pred.shape[-1] // 3
    y       = y_pred[:, :num_nodes]
    y_lower = y_pred[:, num_nodes:2*num_nodes]
    y_upper = y_pred[:, 2*num_nodes:]

    # Pérdida central
    L_center = F.mse_loss(y, y_true)

    # Pérdida asimétrica inferior
    diff_lower = y_true - y_lower
    L_lower = torch.mean(
        torch.maximum(
            (alpha / 2) * diff_lower,
            (alpha / 2 - 1) * diff_lower
        )
    )

    # Pérdida asimétrica superior
    diff_upper = y_true - y_upper
    L_upper = torch.mean(
        torch.maximum(
            (1 - alpha / 2) * diff_upper,
            ((1 - alpha / 2) - 1) * diff_upper
        )
    )

    return L_center + pinball_lambda * (L_lower + L_upper)


class AdaptivePinballLossSoftplus:
    """
    Versión estática de la pérdida Pinball, conservando el nombre por compatibilidad.
    Aplica un multiplicador configurable (pinball_lambda) a la pérdida Pinball.
    """
    def __init__(self, net, warmup_iters=None, calibration_iters=None, alpha=0.1, lambda_min=None, lambda_max=None, pinball_lambda=1.0):
        self.net = net
        self.alpha = alpha
        
        self.step = 0
        self.final_lambda = pinball_lambda
        
        self.net._skip_uncertainty = False
        print(f"\n[Pinball Estático] Inicializado con lambda = {self.final_lambda}")

    def __call__(self, y_true, y_pred):
        self.step += 1

        num_nodes = y_pred.shape[-1] // 3
        y       = y_pred[:, :num_nodes]
        y_lower = y_pred[:, num_nodes:2*num_nodes]
        y_upper = y_pred[:, 2*num_nodes:]

        L_center = F.mse_loss(y, y_true)

        # Calcular Pinball
        diff_lower = y_true - y_lower
        L_lower = torch.mean(torch.maximum((self.alpha / 2) * diff_lower, (self.alpha / 2 - 1) * diff_lower))
        
        diff_upper = y_true - y_upper
        L_upper = torch.mean(torch.maximum((1 - self.alpha / 2) * diff_upper, ((1 - self.alpha / 2) - 1) * diff_upper))

        L_pinball = L_lower + L_upper

        return L_center + self.final_lambda * L_pinball
