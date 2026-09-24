import jax
import jax.numpy as jnp
from jax import jit, vmap, jacrev
import jax.scipy.sparse.linalg
import numpy as np
import h5py
import argparse
import os
from functools import partial

# =============================================================================
# CORE SOLVER LOGIC (2D Darcy Flow)
# =============================================================================

def apply_A(u_interior, K, h):
    """
    Aplica el operador diferencial -\\nabla \\cdot (K \\nabla u)
    u_interior: (N-2, N-2)
    K: (N, N)
    """
    u_full = jnp.pad(u_interior, 1, mode='constant') # (N, N), Dirichlet u=0
    
    # Interpolar K en las interfaces (promedio aritmético)
    K_x = (K[:-1, :] + K[1:, :]) / 2.0 # (N-1, N)
    K_y = (K[:, :-1] + K[:, 1:]) / 2.0 # (N, N-1)
    
    # Flujos
    flux_x = K_x * (u_full[1:, :] - u_full[:-1, :]) / h # (N-1, N)
    flux_y = K_y * (u_full[:, 1:] - u_full[:, :-1]) / h # (N, N-1)
    
    # Divergencia del flujo en los nodos interiores
    div_x = (flux_x[1:, 1:-1] - flux_x[:-1, 1:-1]) / h
    div_y = (flux_y[1:-1, 1:] - flux_y[1:-1, :-1]) / h
    
    return -(div_x + div_y)

@partial(jit, static_argnums=(1,))
def solve_darcy(K, N):
    """
    Resuelve -\\nabla \\cdot (K \\nabla u) = 1 en [0,1]^2 con u=0 en los bordes.
    Utiliza Gradiente Conjugado (CG) soportado por JAX.
    Al usar CG, JAX puede aplicar diferenciación implícita para obtener
    el Jacobiano exacto (jacrev) sin derivar el bucle del solver.
    """
    h = 1.0 / (N - 1)
    b = jnp.ones((N-2, N-2))
    
    def A(u):
        return apply_A(u, K, h)
    
    # Resolvemos el sistema lineal Au = b
    u_interior, info = jax.scipy.sparse.linalg.cg(A, b, maxiter=2000, tol=1e-5)
    
    # Reconstruimos el campo completo (con bordes en 0)
    u_full = jnp.pad(u_interior, 1, mode='constant')
    return u_full

@partial(jit, static_argnums=(1,))
def compute_jacobian_darcy(K, N):
    """
    Extrae la sensibilidad analítica (Jacobiano) \\partial u / \\partial K.
    Retorna un tensor de (N*N, N*N).
    """
    # jacrev calcula la derivada de la salida respecto a K
    # Salida: (N, N) -> (N*N)
    # K: (N, N) -> (N*N)
    def flat_solver(K_in):
        return solve_darcy(K_in, N).flatten()
        
    J = jacrev(flat_solver)(K) # Forma: (N*N, N, N)
    return J.reshape(N*N, N*N)


# =============================================================================
# DATA GENERATION WRAPPER
# =============================================================================

@partial(jit, static_argnums=(1,))
def compute_sensitivity_darcy(K, N):
    """Sensibilidad REDUCIDA: la suma del Jacobiano sobre todas las entradas.

        s_i = sum_j  du_i / dK_j  =  (J @ 1)_i

    Es exactamente la cantidad que consume el analisis de correlaciones
    (abs de la suma sobre las entradas), pero se obtiene con UN solo paso de
    modo directo (jvp en la direccion de unos) en vez de los N*N pasos reversos
    que cuesta la matriz completa, y ocupa N*N numeros por muestra en vez de
    (N*N)^2: 16 KB frente a 67 MB con N=64.
    """
    def flat_solver(K_in):
        return solve_darcy(K_in, N).flatten()

    _, s = jax.jvp(flat_solver, (K,), (jnp.ones_like(K),))
    return s


def generate_grf(key, N, scale=0.5):
    """
    Genera un Campo Aleatorio Gaussiano suave usando serie de Fourier truncada.
    Se expone a un exponencial para garantizar K > 0 (condición de elipticidad).
    """
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    k1, k2 = jax.random.split(key)
    # Frecuencias bajas (3x3)
    amps = jax.random.normal(k1, (3, 3)) * scale
    phases = jax.random.uniform(k2, (3, 3), maxval=2*jnp.pi)
    
    field = jnp.zeros((N, N))
    for i in range(3):
        for j in range(3):
            field += amps[i, j] * jnp.sin(2 * jnp.pi * (i+1) * X + 2 * jnp.pi * (j+1) * Y + phases[i, j])
            
    # Asegurar positividad K > 0
    return jnp.exp(field)

@partial(jit, static_argnums=(1, 2, 3))
def generate_batch(keys, N, compute_jac, reduced=False, grf_scale=0.5):
    K_batch = vmap(generate_grf, in_axes=(0, None, None))(keys, N, grf_scale)
    u_batch = vmap(solve_darcy, in_axes=(0, None))(K_batch, N)

    # Para JAX, es mejor usar un if de Python estático o separar las funciones,
    # pero aquí podemos mapear compute_jacobian_darcy de forma segura.
    if compute_jac:
        fn = compute_sensitivity_darcy if reduced else compute_jacobian_darcy
        J_batch = vmap(fn, in_axes=(0, None))(K_batch, N)
        return K_batch, u_batch, J_batch
    else:
        return K_batch, u_batch, jnp.zeros((len(keys), 0))

def generate_darcy_dataset(num_samples, N, compute_jacobian=True, batch_size=10, seed_offset=42,
                           jacobian_mode="full", grf_scale=0.5):
    x = jnp.linspace(0, 1, N)
    y = jnp.linspace(0, 1, N)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    trunk_inputs = jnp.stack([X.flatten(), Y.flatten()], axis=-1)
    
    all_K, all_u, all_J = [], [], []
    samples_collected = 0
    
    print(f"Generando {num_samples} muestras de Darcy Flow (Jacobiano: {compute_jacobian}"
          f"{'/' + jacobian_mode if compute_jacobian else ''}, seed_offset: {seed_offset}, grf_scale: {grf_scale})...")
    
    while samples_collected < num_samples:
        current_batch = min(batch_size, num_samples - samples_collected)
        keys = jax.random.split(jax.random.PRNGKey(samples_collected + seed_offset), current_batch)
        
        K_batch, u_batch, J_batch = generate_batch(keys, N, compute_jacobian,
                                                   jacobian_mode == "sum", grf_scale)
        
        all_K.append(np.array(K_batch.reshape(current_batch, -1)))
        all_u.append(np.array(u_batch.reshape(current_batch, -1)))
        
        if compute_jacobian:
            all_J.append(np.array(J_batch))
            
        samples_collected += current_batch
        print(f"Progreso: {samples_collected}/{num_samples}")

    return {
        "branch_inputs": np.concatenate(all_K, axis=0),
        "trunk_inputs": np.array(trunk_inputs),
        "targets": np.concatenate(all_u, axis=0),
        **({"jacobian_reference": np.concatenate(all_J, axis=0)} if compute_jacobian else {})
    }

def main():
    parser = argparse.ArgumentParser(description="Generador Darcy Flow 2D + Sensibilidad Física JAX")
    parser.add_argument("--num_samples", type=int, default=100, help="Número de simulaciones")
    parser.add_argument("--N", type=int, default=31, help="Puntos espaciales por dimensión (NxN)")
    parser.add_argument("--output_path", type=str, default="data/darcy_sensitivity.h5", help="Ruta de salida")
    parser.add_argument("--batch_size", type=int, default=10, help="Tamaño del lote para vmap")
    parser.add_argument("--skip_jacobian", action="store_true", help="Saltar el cálculo del Jacobiano")
    parser.add_argument("--seed_offset", type=int, default=42, help="Offset de semilla para evitar colisiones entre train/test")
    parser.add_argument("--jacobian_mode", type=str, choices=["full", "sum"], default="full",
                        help="full: matriz du/dK completa (N*N x N*N, 67 MB por muestra con N=64). "
                             "sum: su suma sobre las entradas (N*N por muestra), que es lo que usa "
                             "el analisis de correlaciones.")
    parser.add_argument("--grf_scale", type=float, default=0.5,
                        help="Desviacion estandar de amplitudes del GRF (default: 0.5, OOD: >0.5)")
    
    args = parser.parse_args()
    
    # Path relative to execution context
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    data = generate_darcy_dataset(
        args.num_samples, args.N, 
        compute_jacobian=not args.skip_jacobian, batch_size=args.batch_size,
        seed_offset=args.seed_offset, jacobian_mode=args.jacobian_mode,
        grf_scale=args.grf_scale
    )
    
    with h5py.File(args.output_path, "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v, compression="gzip")
        
        f.attrs["equation"] = "darcy_2d"
        f.attrs["N"] = args.N
        f.attrs["description"] = "Darcy Flow 2D dataset with parametric sensitivity du/dK computed via JAX implicit AD."
        f.attrs["jacobian_mode"] = args.jacobian_mode

    print(f"Dataset guardado en {args.output_path}")

if __name__ == "__main__":
    main()
