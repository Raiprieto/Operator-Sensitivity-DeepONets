import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import h5py
import argparse
import os
from tqdm import tqdm

from functools import partial

# =============================================================================
# CORE SOLVER LOGIC (Adapted from PDEBench)
# =============================================================================

def VLlimiter(a, b, c, alpha=2.0):
    """
    Limitador de Van Leer exacto. 
    Nota: Las funciones jnp.sign y jnp.abs no son diferenciables en cero. 
    Si se intenta usar Diferenciación Automática (JAX AD), generarán NaNs.
    """
    return (
        jnp.sign(c)
        * (0.5 + 0.5 * jnp.sign(a * b))
        * jnp.minimum(alpha * jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    )

@partial(jit, static_argnums=(1,))
def limiting(u, nx, if_second_order=1.0):
    # u is size nx+4
    du_L = u[1 : nx + 3] - u[0 : nx + 2]
    du_R = u[2 : nx + 4] - u[1 : nx + 3]
    du_M = (u[2 : nx + 4] - u[0 : nx + 2]) * 0.5
    gradu = VLlimiter(du_L, du_R, du_M) * if_second_order
    
    uL = jnp.zeros_like(u)
    uR = jnp.zeros_like(u)
    
    # left and right values at cell centers? No, at interfaces.
    # PDEBench: uL is the value at the LEFT interface of the cell
    uL = uL.at[1 : nx + 3].set(u[1 : nx + 3] - 0.5 * gradu)
    uR = uR.at[1 : nx + 3].set(u[1 : nx + 3] + 0.5 * gradu)
    return uL, uR

@partial(jit, static_argnums=(1,))
def bc_periodic(u, nx):
    _u = jnp.zeros(nx + 4)
    _u = _u.at[2 : nx + 2].set(u)
    _u = _u.at[0:2].set(u[-2:])
    _u = _u.at[nx + 2 : nx + 4].set(u[0:2])
    return _u

@partial(jit, static_argnums=(1,))
def flux(u, nx, dx_inv, epsilon):
    _u = bc_periodic(u, nx)
    uL_ghost, uR_ghost = limiting(_u, nx)
    
    # Values at interface i+1/2
    # uR_ghost[i] is right value of cell i
    # uL_ghost[i+1] is left value of cell i+1
    uR_int = uR_ghost[1 : nx + 2]
    uL_int = uL_ghost[2 : nx + 3]
    
    fR = 0.5 * uR_int**2
    fL = 0.5 * uL_int**2
    
    # Upwind flux
    f_upwd = 0.5 * (fR + fL - 0.5 * jnp.abs(uL_int + uR_int) * (uL_int - uR_int))
    
    # Diffusion flux at interface
    f_diff = -epsilon * (_u[2 : nx + 3] - _u[1 : nx + 2]) * dx_inv
    
    return f_upwd + f_diff

@partial(jit, static_argnums=(1,))
def update(u, nx, dx_inv, dt, epsilon):
    f = flux(u, nx, dx_inv, epsilon)
    u_new = u - dt * dx_inv * (f[1 : nx + 1] - f[0 : nx])
    return u_new

@partial(jit, static_argnums=(1,))
def step_fn(u, nx, dx_inv, dt, epsilon):
    u_mid = update(u, nx, dx_inv, dt * 0.5, epsilon)
    f_mid = flux(u_mid, nx, dx_inv, epsilon)
    u_corr = u - dt * dx_inv * (f_mid[1 : nx + 1] - f_mid[0 : nx])
    return u_corr

@partial(jit, static_argnums=(1, 4))
def solve_burgers(u0, nx, dx, dt_output, nt_output, epsilon):
    """
    Simulates Burgers equation with a FIXED number of sub-steps for JAX differentiability.
    """
    dx_inv = 1.0 / dx
    
    # Use a fixed large number of sub-steps to ensure stability across all nu in [0.001, 0.1]
    # 40 steps is safe for dx=1/128 and dt=0.01
    n_substeps = 40
    dt_inner = dt_output / n_substeps
    
    def body_fn(u_curr, _):
        def sub_step_loop(i, u):
            return step_fn(u, nx, dx_inv, dt_inner, epsilon)
        
        # Use fori_loop with static n_substeps
        u_next = lax.fori_loop(0, n_substeps, sub_step_loop, u_curr)
        return u_next, u_next
    
    _, rest_history = lax.scan(body_fn, u0, jnp.arange(nt_output - 1))
    
    full_history = jnp.vstack([u0[None, :], rest_history])
    return full_history

# =============================================================================
# DATA GENERATION WRAPPER
# =============================================================================

@partial(jit, static_argnums=(1, 4))
def solve_burgers_batch(u0_batch, nx, dx, dt, nt, nu_batch):
    return vmap(solve_burgers, in_axes=(0, None, None, None, None, 0))(
        u0_batch, nx, dx, dt, nt, nu_batch
    )

def generate_burgers_dataset(num_samples, nx, nt, L, T, nu_range, compute_jacobian=True, batch_size=50, base_seed=0):
    """
    Generates N samples of Burgers 1D + Sensitivity using batch processing and quality filtering.
    """
    dx = L / nx
    dt_out = T / nt
    x = jnp.linspace(0, L, nx, endpoint=False)
    t = jnp.linspace(0, T, nt)
    
    X, T_grid = jnp.meshgrid(x, t, indexing='ij')
    trunk_inputs = jnp.stack([X.flatten(), T_grid.flatten()], axis=-1)
    
    all_branch, all_targets, all_jacobians, all_nus = [], [], [], []
    samples_collected = 0
    # base_seed=0 reproduce los datasets originales. Para generar conjuntos
    # de test usar un base_seed lejano (p. ej. 1_000_000): con seed_offset=0
    # fijo, dos corridas producian las MISMAS muestras (leakage train/test).
    seed_offset = base_seed
    
    if compute_jacobian:
        print("Aviso: Calculando Jacobiano vía Diferencias Finitas Centrales (delta_nu = 1e-4) para evitar NaNs en el limitador de choques.")

    print(f"Generando {num_samples} muestras válidas (Jacobiano: {compute_jacobian})...")
    
    while samples_collected < num_samples:
        current_batch = min(batch_size, num_samples - samples_collected)
        keys = jax.random.split(jax.random.PRNGKey(samples_collected + seed_offset), current_batch)
        seed_offset += current_batch # Increment offset to avoid repeating seeds if we retry
        
        def gen_u0(key):
            k1, k2 = jax.random.split(key)
            amps = jax.random.uniform(k1, (5,), minval=-1.0, maxval=1.0)
            phases = jax.random.uniform(k2, (5,), minval=0, maxval=2*jnp.pi)
            u0 = jnp.zeros(nx)
            for i in range(5):
                u0 += amps[i] * jnp.sin(2 * jnp.pi * (i+1) * x / L + phases[i])
            return u0

        u0_batch = vmap(gen_u0)(keys)
        nu_batch = jax.random.uniform(jax.random.PRNGKey(samples_collected + seed_offset + 10000), (current_batch,), 
                                     minval=nu_range[0], maxval=nu_range[1])
        
        # Solve
        u_sol_batch = solve_burgers_batch(u0_batch, nx, dx, dt_out, nt, nu_batch)
        
        # Filter NaNs
        mask = ~jnp.any(jnp.isnan(u_sol_batch.reshape(current_batch, -1)), axis=1)
        valid_indices = jnp.where(mask)[0]
        num_valid = len(valid_indices)
        
        if num_valid > 0:
            u0_valid = u0_batch[valid_indices]
            u_sol_valid = u_sol_batch[valid_indices]
            nu_valid = nu_batch[valid_indices]
            
            # CRITICAL: Transpose to (N, nx, nt) to match trunk_inputs order (x first, then t)
            u_sol_flat = u_sol_valid.transpose(0, 2, 1).reshape(num_valid, -1)
            
            if compute_jacobian:
                # Implementación de Diferencias Finitas Centrales (Fidelidad 100% de la PDE original)
                # J \approx (u(nu + delta) - u(nu - delta)) / (2 * delta)
                delta_nu = 1e-4
                nu_plus = nu_valid + delta_nu
                nu_minus = nu_valid - delta_nu
                
                u_plus = solve_burgers_batch(u0_valid, nx, dx, dt_out, nt, nu_plus)
                u_minus = solve_burgers_batch(u0_valid, nx, dx, dt_out, nt, nu_minus)
                
                u_jac_valid = (u_plus - u_minus) / (2.0 * delta_nu)
                
                # Alinear tensores igual que en la data forward
                u_jac_flat = u_jac_valid.transpose(0, 2, 1).reshape(num_valid, -1)
                all_jacobians.append(np.array(u_jac_flat))
            
            all_branch.append(np.array(u0_valid))
            all_targets.append(np.array(u_sol_flat))
            all_nus.append(np.array(nu_valid))
            samples_collected += num_valid
            
        print(f"Progreso: {samples_collected}/{num_samples} (Descartados en este lote: {current_batch - num_valid})")

    return {
        "branch_inputs": np.concatenate(all_branch, axis=0)[:num_samples],
        "trunk_inputs": np.array(trunk_inputs),
        "targets": np.concatenate(all_targets, axis=0)[:num_samples],
        "params_coefficient": np.concatenate(all_nus, axis=0)[:num_samples],
        **({"jacobian_reference": np.concatenate(all_jacobians, axis=0)[:num_samples]} if compute_jacobian else {})
    }

def main():
    parser = argparse.ArgumentParser(description="Generador de datos Burgers 1D + Sensibilidad Física (JAX)")
    parser.add_argument("--num_samples", type=int, default=100, help="Número de simulaciones")
    parser.add_argument("--nx", type=int, default=128, help="Puntos espaciales")
    parser.add_argument("--nt", type=int, default=100, help="Pasos de tiempo")
    parser.add_argument("--L", type=float, default=1.0, help="Longitud del dominio")
    parser.add_argument("--T", type=float, default=1.0, help="Tiempo final")
    parser.add_argument("--nu_min", type=float, default=0.001, help="Viscosidad mínima")
    parser.add_argument("--nu_max", type=float, default=0.1, help="Viscosidad máxima")
    parser.add_argument("--output_path", type=str, default="data/burgers_sensitivity.h5", help="Ruta de salida")
    parser.add_argument("--batch_size", type=int, default=100, help="Tamaño del lote para vmap")
    parser.add_argument("--skip_jacobian", action="store_true", help="Saltar el cálculo del Jacobiano")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Semilla base. 0 reproduce los datos originales; usar un valor lejano (p. ej. 1000000) para test")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    data = generate_burgers_dataset(
        args.num_samples, args.nx, args.nt, args.L, args.T, (args.nu_min, args.nu_max),
        compute_jacobian=not args.skip_jacobian, batch_size=args.batch_size,
        base_seed=args.seed_offset
    )
    
    with h5py.File(args.output_path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")
        
        f.attrs["equation"] = "burgers_1d"
        f.attrs["nx"] = args.nx
        f.attrs["nt"] = args.nt
        f.attrs["L"] = args.L
        f.attrs["T"] = args.T
        f.attrs["description"] = "Burgers 1D dataset with parametric sensitivity du/dnu computed via JAX AD."

    print(f"Dataset guardado en {args.output_path}")

if __name__ == "__main__":
    main()
