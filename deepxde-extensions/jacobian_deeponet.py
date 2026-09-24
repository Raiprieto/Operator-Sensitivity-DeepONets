import torch
import torch.nn as nn
import torch.nn.functional as F
import deepxde as dde

def create_jacobian_deeponet(is_cartesian=False):
    BaseClass = dde.nn.DeepONetCartesianProd if is_cartesian else dde.nn.DeepONet
    
    class JacobianDeepONet(BaseClass):
        def __init__(self, layer_sizes_branch, layer_sizes_trunk, 
                     activation, kernel_initializer, jacobian_type="parameter", sigma=0.05):
            super().__init__(
                layer_sizes_branch=layer_sizes_branch,
                layer_sizes_trunk=layer_sizes_trunk,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            self.sigma = sigma
            self.jacobian_type = jacobian_type
            
            # Pesos asimétricos entrenables
            # Inicializados en 0 para que softplus(0) ≈ 0.69
            # y las bandas iniciales sean simétricas y pequeñas
            self.w_lower = nn.Parameter(torch.zeros(1))
            self.w_upper = nn.Parameter(torch.zeros(1))

        def _compute_uncertainty(self, y, u, x, is_cartesian):
            if is_cartesian and self.jacobian_type == "parameter":
                # Descomposición B(u) · T(x)^T respetando la API de DeepXDE
                B = self.branch(u)   # [batch, K]
                # self.b se añade al final de y, su derivada respecto a u es 0.
                
                T_raw = self.trunk(x)    # [N_nodes, K]
                T = self.activation_trunk(T_raw) if hasattr(self, "activation_trunk") else T_raw
                
                K = B.shape[-1]

                # Jacobiano del branch: Vectorizado con jacrev (PyTorch 2.0+) para máxima velocidad en GPU
                try:
                    from torch.func import vmap, jacrev
                    
                    def _compute_branch_single(u_s):
                        return self.branch(u_s.unsqueeze(0)).squeeze(0)
                    
                    # vmap vectoriza el batch, jacrev saca la jacobiana de [K] respecto a [N_in]
                    J_branch = vmap(jacrev(_compute_branch_single))(u)
                except ImportError:
                    # Fallback al lento bucle for de Python si PyTorch es antiguo
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
                        J_branch.append(grad_k)  # [batch, N_in]
                    J_branch = torch.stack(J_branch, dim=1)  # [batch, K, N_in]

                # Varianza pointwise exacta mediante Matriz de Gram (Optimización para K pequeño):
                # En lugar de hacer un bucle for de N_in iteraciones (que causaba OOM en el backward pass
                # al acumular el grafo computacional para N_in=4225), aprovechamos que K=64.
                # Calculamos la matriz de Gram latente G = J * J^T de tamaño [batch, K, K].
                # Y luego obtenemos la varianza proyectando con T: V = diag(T * G * T^T).
                
                # 1. Matriz de Gram Latente [batch, K, K]
                G = torch.bmm(J_branch, J_branch.transpose(1, 2))
                
                # 2. Proyección intermedia [batch, N_nodes, K]
                # (Esta operación pesa ~138 MB, perfectamente manejable por la GPU de 6GB)
                TG = torch.einsum('nk,bkm->bnm', T, G)
                
                # Varianza final [batch, N_nodes]
                variance = torch.einsum('bnm,nm->bn', TG, T)
                
                variance = variance * (self.sigma**2)

            elif self.jacobian_type == "parameter":
                # Pointwise: gradiente directo por nodo (Fallback para redes no cartesianas)
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

            return torch.sqrt(variance + 1e-8)  # [batch, N_nodes]

        def forward(self, inputs):
            u, x = inputs

            if not u.requires_grad:
                u.requires_grad_(True)
            if not x.requires_grad:
                x.requires_grad_(True)

            # Predicción central
            y = super().forward((u, x))

            #Incertidumbre estructurada por el Jacobiano
            uncertainty = self._compute_uncertainty(y, u, x, is_cartesian)

            # Normalización del Jacobiano: separamos TOPOLOGÍA (forma espacial)
            # de MAGNITUD (escala absoluta). Esto evita que el optimizador empuje
            # w hacia -inf cuando ||J|| es muy grande y el error muy pequeño.
            # .detach() evita que el gradiente fluya por la normalización.
            unc_scale = uncertainty.mean().detach() + 1e-8
            unc_shape = uncertainty / unc_scale  # media ≈ 1.0

            # exp(w) en lugar de softplus(w): la optimización ocurre en espacio
            # logarítmico, donde ajustes de escala son multiplicativos y más
            # estables. Con w=0 inicializado, exp(0)=1.0 → bandas iniciales = σ * unc_shape.
            delta_lower = torch.exp(self.w_lower) * unc_shape
            delta_upper = torch.exp(self.w_upper) * unc_shape

            y_lower = y - delta_lower
            y_upper = y + delta_upper

            return torch.cat([y, y_lower, y_upper], dim=-1)

    return JacobianDeepONet


def pinball_loss(y_true, y_pred):
    alpha = 0.1  # 95% de cobertura objetivo

    num_nodes = y_pred.shape[-1] // 3
    y       = y_pred[:, :num_nodes]
    y_lower = y_pred[:, num_nodes:2*num_nodes]
    y_upper = y_pred[:, 2*num_nodes:]

    # Pérdida central
    L_center = F.mse_loss(y, y_true)

    # Pérdida asimétrica inferior (cuantil alpha/2)
    diff_lower = y_true - y_lower
    L_lower = torch.mean(
        torch.maximum(
            (alpha / 2) * diff_lower,
            (alpha / 2 - 1) * diff_lower
        )
    )

    # Pérdida asimétrica superior (cuantil 1 - alpha/2)
    diff_upper = y_true - y_upper
    L_upper = torch.mean(
        torch.maximum(
            (1 - alpha / 2) * diff_upper,
            ((1 - alpha / 2) - 1) * diff_upper
        )
    )

    return L_center + L_lower + L_upper