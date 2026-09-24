import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
import numpy as np
import h5py
import argparse
import os
from functools import partial

# =============================================================================
# CORE SOLVER LOGIC (2D Compressible Navier-Stokes)
# =============================================================================

def get_fluxes(U, gamma):
    """
    Calcula los flujos F y G de Euler para las ecuaciones de Navier-Stokes.
    U = [rho, rho*u, rho*v, E]
    """
    rho = U[0]
    rhou = U[1]
    rhov = U[2]
    E = U[3]
    
    u = rhou / jnp.maximum(rho, 1e-5)
    v = rhov / jnp.maximum(rho, 1e-5)
    
    p = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    p = jnp.maximum(p, 1e-5) # Presión siempre positiva
    
    F = jnp.stack([
        rhou,
        rhou * u + p,
        rhou * v,
        u * (E + p)
    ])
    
    G = jnp.stack([
        rhov,
        rhou * v,
        rhov * v + p,
        v * (E + p)
    ])
    
    return F, G, u, v, p

def bc_periodic(U):
    """Aplica condiciones de borde periódicas a la malla."""
    # U shape: (4, nx, nx)
    # Pad to (4, nx+2, nx+2)
    return jnp.pad(U, ((0,0), (1,1), (1,1)), mode='wrap')

def maccormack_step(U, gamma, mu, dx, dy, dt):
    """
    Paso de integración explícita usando el esquema predictor-corrector de MacCormack.
    Incluye viscosidad explícita para emular comportamiento Navier-Stokes y estabilizar.
    """
    U_pad = bc_periodic(U)
    F_pad, G_pad, u_pad, v_pad, _ = get_fluxes(U_pad, gamma)
    
    # --- Predictor (Forward differences) ---
    dF_dx_fwd = (F_pad[:, 2:, 1:-1] - F_pad[:, 1:-1, 1:-1]) / dx
    dG_dy_fwd = (G_pad[:, 1:-1, 2:] - G_pad[:, 1:-1, 1:-1]) / dy
    
    # Términos viscosos (Diferencias centrales de 2do orden)
    d2u_dx2 = (u_pad[2:, 1:-1] - 2*u_pad[1:-1, 1:-1] + u_pad[:-2, 1:-1]) / (dx**2)
    d2u_dy2 = (u_pad[1:-1, 2:] - 2*u_pad[1:-1, 1:-1] + u_pad[1:-1, :-2]) / (dy**2)
    d2v_dx2 = (v_pad[2:, 1:-1] - 2*v_pad[1:-1, 1:-1] + v_pad[:-2, 1:-1]) / (dx**2)
    d2v_dy2 = (v_pad[1:-1, 2:] - 2*v_pad[1:-1, 1:-1] + v_pad[1:-1, :-2]) / (dy**2)
    
    visc_u = mu * (d2u_dx2 + d2u_dy2)
    visc_v = mu * (d2v_dx2 + d2v_dy2)
    Visc = jnp.stack([jnp.zeros_like(visc_u), visc_u, visc_v, jnp.zeros_like(visc_u)])
    
    U_star = U - dt * (dF_dx_fwd + dG_dy_fwd) + dt * Visc
    
    # --- Corrector (Backward differences) ---
    U_star_pad = bc_periodic(U_star)
    F_star_pad, G_star_pad, u_star_pad, v_star_pad, _ = get_fluxes(U_star_pad, gamma)
    
    dF_dx_bwd = (F_star_pad[:, 1:-1, 1:-1] - F_star_pad[:, :-2, 1:-1]) / dx
    dG_dy_bwd = (G_star_pad[:, 1:-1, 1:-1] - G_star_pad[:, 1:-1, :-2]) / dy
    
    d2u_dx2_star = (u_star_pad[2:, 1:-1] - 2*u_star_pad[1:-1, 1:-1] + u_star_pad[:-2, 1:-1]) / (dx**2)
    d2u_dy2_star = (u_star_pad[1:-1, 2:] - 2*u_star_pad[1:-1, 1:-1] + u_star_pad[1:-1, :-2]) / (dy**2)
    d2v_dx2_star = (v_star_pad[2:, 1:-1] - 2*v_star_pad[1:-1, 1:-1] + v_star_pad[:-2, 1:-1]) / (dx**2)
    d2v_dy2_star = (v_star_pad[1:-1, 2:] - 2*v_star_pad[1:-1, 1:-1] + v_star_pad[1:-1, :-2]) / (dy**2)
    
    visc_u_star = mu * (d2u_dx2_star + d2u_dy2_star)
    visc_v_star = mu * (d2v_dx2_star + d2v_dy2_star)
    Visc_star = jnp.stack([jnp.zeros_like(visc_u_star), visc_u_star, visc_v_star, jnp.zeros_like(visc_u_star)])
    
    U_new = 0.5 * (U + U_star - dt * (dF_dx_bwd + dG_dy_bwd) + dt * Visc_star)
    
    return U_new

def clip_physics(U, gamma):
    """Estabilización numérica para prevenir divergencia no-física."""
    rho = jnp.maximum(U[0], 1e-3)
    rhou = U[1]
    rhov = U[2]
    
    E_min = 0.5 * (rhou**2 + rhov**2) / rho + 1e-3 / (gamma - 1.0)
    E = jnp.maximum(U[3], E_min)
    
    return jnp.stack([rho, rhou, rhov, E])

@partial(jit, static_argnums=(3,4,5,6))
def solve_ns_2d(rho0, gamma, mu, nx=64, dt=1e-4, nt_out=20, n_substeps=50):
    """
    Resuelve Navier-Stokes 2D por 'nt_out' snapshots.
    Retorna la densidad a lo largo del tiempo.
    """
    dx = 1.0 / nx
    dy = 1.0 / nx
    
    # Relación Isentrópica inicial: p0 = rho0^gamma
    p0 = rho0 ** gamma
    E0 = p0 / (gamma - 1.0)
    
    U0 = jnp.stack([
        rho0,
        jnp.zeros_like(rho0),
        jnp.zeros_like(rho0),
        E0
    ])
    
    def step_loop(i, U):
        U = maccormack_step(U, gamma, mu, dx, dy, dt)
        U = clip_physics(U, gamma)
        return U
        
    def body_fn(U, _):
        U_next = jax.lax.fori_loop(0, n_substeps, step_loop, U)
        # Solo retornamos la densidad (rho) como target
        return U_next, U_next[0]
        
    _, rho_history = jax.lax.scan(body_fn, U0, jnp.arange(nt_out - 1))
    
    # Agregar instante inicial t=0
    full_rho_history = jnp.vstack([rho0[None, :, :], rho_history])
    return full_rho_history

# =============================================================================
# DATA GENERATION WRAPPER
# =============================================================================

@partial(jit, static_argnums=(3, 4, 5, 6))
def solve_ns_batch(rho0_batch, gamma_batch, mu_batch, nx, dt, nt_out, n_substeps):
    return vmap(solve_ns_2d, in_axes=(0, 0, 0, None, None, None, None))(
        rho0_batch, gamma_batch, mu_batch, nx, dt, nt_out, n_substeps
    )

def generate_ns_dataset(num_samples, nx, nt_out, T, gamma_range, mu_range, compute_jacobian=True, batch_size=20):
    """Genera lote de datos HDF5 de simulaciones Navier-Stokes con sensibilidades analíticas vía JVP/FD."""
    dx = 1.0 / nx
    dt_out = T / nt_out
    n_substeps = 50
    dt_inner = dt_out / n_substeps # dt = 0.0005 para T=0.5, muy estable
    
    x = jnp.linspace(0, 1, nx, endpoint=False)
    y = jnp.linspace(0, 1, nx, endpoint=False)
    t = jnp.linspace(0, T, nt_out)
    
    X, Y, T_grid = jnp.meshgrid(x, y, t, indexing='ij')
    trunk_inputs = jnp.stack([X.flatten(), Y.flatten(), T_grid.flatten()], axis=-1)
    
    all_branch, all_targets, all_jacobians, all_params = [], [], [], []
    samples_collected = 0
    seed_offset = 0
    
    if compute_jacobian:
        print("Aviso: Calculando Jacobiano vía Diferencias Finitas Centrales (delta_gamma = 1e-4)")

    print(f"Generando {num_samples} muestras válidas (Jacobiano: {compute_jacobian})...")
    
    while samples_collected < num_samples:
        current_batch = min(batch_size, num_samples - samples_collected)
        keys = jax.random.split(jax.random.PRNGKey(samples_collected + seed_offset), current_batch)
        seed_offset += current_batch
        
        def gen_rho0_vmap(key):
            k1, k2 = jax.random.split(key)
            amps = jax.random.normal(k1, (4, 4)) * 0.5
            phases = jax.random.uniform(k2, (4, 4), maxval=2*jnp.pi)
            
            x_idx = jnp.linspace(0, 1, nx, endpoint=False)
            y_idx = jnp.linspace(0, 1, nx, endpoint=False)
            X_idx, Y_idx = jnp.meshgrid(x_idx, y_idx, indexing='ij')
            
            field = jnp.zeros((nx, nx))
            for i in range(4):
                for j in range(4):
                    field += amps[i, j] * jnp.sin(2 * jnp.pi * (i+1) * X_idx + 2 * jnp.pi * (j+1) * Y_idx + phases[i, j])
            
            field = field - jnp.mean(field)
            rho = jnp.exp(field)
            rho_min = jnp.min(rho)
            rho_max = jnp.max(rho)
            rho_scaled = 0.5 + 2.0 * (rho - rho_min) / (rho_max - rho_min + 1e-6)
            return rho_scaled
            
        rho0_batch = vmap(gen_rho0_vmap)(keys)
        
        rng_gamma = jax.random.PRNGKey(samples_collected + seed_offset + 10000)
        rng_mu = jax.random.PRNGKey(samples_collected + seed_offset + 20000)
        gamma_batch = jax.random.uniform(rng_gamma, (current_batch,), minval=gamma_range[0], maxval=gamma_range[1])
        mu_batch = jax.random.uniform(rng_mu, (current_batch,), minval=mu_range[0], maxval=mu_range[1])
        
        # Simulación principal
        rho_sol_batch = solve_ns_batch(rho0_batch, gamma_batch, mu_batch, nx, dt_inner, nt_out, n_substeps)
        
        # Filtrar muestras inestables (NaNs)
        mask = ~jnp.any(jnp.isnan(rho_sol_batch.reshape(current_batch, -1)), axis=1)
        valid_indices = jnp.where(mask)[0]
        num_valid = len(valid_indices)
        
        if num_valid > 0:
            rho0_valid = rho0_batch[valid_indices]
            rho_sol_valid = rho_sol_batch[valid_indices]
            gamma_valid = gamma_batch[valid_indices]
            mu_valid = mu_batch[valid_indices]
            
            # Transponer para que coincida con trunk_inputs (X, Y, T_grid)
            # rho_sol_valid: (batch, t, x, y) -> (batch, x, y, t)
            rho_sol_flat = rho_sol_valid.transpose(0, 2, 3, 1).reshape(num_valid, -1)
            
            if compute_jacobian:
                delta_gamma = 1e-4
                gamma_plus = gamma_valid + delta_gamma
                gamma_minus = gamma_valid - delta_gamma
                
                rho_plus = solve_ns_batch(rho0_valid, gamma_plus, mu_valid, nx, dt_inner, nt_out, n_substeps)
                rho_minus = solve_ns_batch(rho0_valid, gamma_minus, mu_valid, nx, dt_inner, nt_out, n_substeps)
                
                # Sensibilidad central
                rho_jac_valid = (rho_plus - rho_minus) / (2.0 * delta_gamma)
                rho_jac_flat = rho_jac_valid.transpose(0, 2, 3, 1).reshape(num_valid, -1)
                all_jacobians.append(np.array(rho_jac_flat))
            
            # Guardar resultados y aplanar densidad inicial
            all_branch.append(np.array(rho0_valid.reshape(num_valid, -1)))
            all_targets.append(np.array(rho_sol_flat))
            
            params = jnp.stack([gamma_valid, mu_valid], axis=1)
            all_params.append(np.array(params))
            samples_collected += num_valid
            
        print(f"Progreso: {samples_collected}/{num_samples} (Descartados en lote: {current_batch - num_valid})")

    return {
        "branch_inputs": np.concatenate(all_branch, axis=0)[:num_samples],
        "trunk_inputs": np.array(trunk_inputs),
        "targets": np.concatenate(all_targets, axis=0)[:num_samples],
        "params_coefficient": np.concatenate(all_params, axis=0)[:num_samples],
        **({"jacobian_reference": np.concatenate(all_jacobians, axis=0)[:num_samples]} if compute_jacobian else {})
    }

def main():
    parser = argparse.ArgumentParser(description="Generador Navier-Stokes 2D Compresible (JAX)")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--gamma_min", type=float, default=1.2)
    parser.add_argument("--gamma_max", type=float, default=2.0)
    parser.add_argument("--mu_min", type=float, default=0.001)
    parser.add_argument("--mu_max", type=float, default=0.01)
    parser.add_argument("--output_path", type=str, default="data/ns2d_sensitivity.h5")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--skip_jacobian", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    data = generate_ns_dataset(
        args.num_samples, args.nx, args.nt, args.T, 
        (args.gamma_min, args.gamma_max), (args.mu_min, args.mu_max),
        compute_jacobian=not args.skip_jacobian, batch_size=args.batch_size
    )
    
    with h5py.File(args.output_path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")
        
        f.attrs["equation"] = "navier_stokes_2d"
        f.attrs["nx"] = args.nx
        f.attrs["nt"] = args.nt
        f.attrs["T"] = args.T
        f.attrs["description"] = "Navier-Stokes 2D compressible dataset. branch_inputs=rho0, params=[gamma, mu], target=rho, jacobian=d_rho/d_gamma via FD."

    print(f"Dataset guardado en {args.output_path}")

if __name__ == "__main__":
    main()
