"""
mc_benchmark.py

Script unificado para evaluar la Cuantificacion de Incertidumbre (UQ) de
cualquier modelo DeepONet (Baseline, Jacobian, Vanilla Pinball) mediante
Monte Carlo: se perturba la entrada N veces con ruido gaussiano y se construyen
bandas de confianza a partir de los percentiles de las predicciones perturbadas.

Metricas entregadas (mismas que abstract_benchmark.py + timing):
- MSE Prediccion Central
- Cobertura Empirica (percentiles 5%-95%)
- MPIW (Mean Prediction Interval Width)
- Correlacion (Pearson/Spearman) ancho vs |Sensibilidad| (si Jacobian reference existe)
- Correlacion (Pearson/Spearman) ancho vs |Error|
- Tiempo promedio de MC por muestra (N simulaciones para un punto de incertidumbre)

Hiperparametros MC:
- alpha: desviacion de la perturbacion como fraccion del parametro base
- n_mc: numero de simulaciones por muestra
- param_idx: indice del parametro a perturbar (si es escalar embebido en la branch)
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
import h5py

# ── Configuracion de Paths ──────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [REPO_ROOT,
          os.path.join(REPO_ROOT, "deepxde-extensions"),
          os.path.join(REPO_ROOT, "benchmarks")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import deepxde as dde
from jacobian_deeponet import create_jacobian_deeponet
from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus
from vanilla_pinball_deeponet import create_vanilla_pinball_deeponet


def parse_args():
    """Analiza los argumentos CLI para el benchmark MC."""
    parser = argparse.ArgumentParser(description="Monte Carlo Benchmark Evaluator")

    # Rutas
    parser.add_argument("--data_path", type=str, required=True,
                        help="Ruta al dataset de prueba (.h5).")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directorio del modelo (.pt y scalers).")
    parser.add_argument("--model_stem", type=str, required=True,
                        help="Nombre base del modelo (ej. burgers_C_DeepThin).")
    parser.add_argument("--model_iter", type=int, default=-1,
                        help="Iteracion del checkpoint. -1 = auto-detect ultimo.")

    # Tipo de modelo
    parser.add_argument("--model_type", type=str, default="baseline",
                        choices=["baseline", "jacobian", "vanilla_pinball"],
                        help="Tipo de modelo a cargar.")

    # Arquitectura
    parser.add_argument("--branch_layers", type=int, nargs="+", required=True,
                        help="Arquitectura de la Branch (ej. 129 256 256 256).")
    parser.add_argument("--trunk_layers", type=int, nargs="+", required=True,
                        help="Arquitectura de la Trunk (ej. 2 256 256 256).")
    parser.add_argument("--activation", type=str, default="swish",
                        help="Funcion de activacion.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Sigma escalar (para modelos Jacobian/Vanilla Pinball).")
    parser.add_argument("--variance_activation", type=str, choices=["exp", "softplus"],
                        default="softplus", help="Activacion de varianza (solo Jacobian).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Hiperparametros MC
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Desviacion de la perturbacion como fraccion del valor base.")
    parser.add_argument("--n_mc", type=int, default=1000,
                        help="Numero de simulaciones MC por muestra.")
    parser.add_argument("--param_idx", type=int, default=None,
                        help="Indice del parametro a perturbar (ej. 128 para viscosidad en Burgers)."
                             " Si no se especifica, se perturba todo el vector de entrada.")

    parser.add_argument("--infer_batch_size", type=int, default=0,
                        help="Batch size para inferencia MC. 0 = sin batching.")

    return parser.parse_args()


def load_data(data_path, expected_branch_dim):
    """Lee los tensores HDF5, concatenando parametros anexos si corresponde.

    Args:
        data_path: Ruta al .h5 de testing.
        expected_branch_dim: Dimension esperada para la branch (branch_layers[0]).

    Returns:
        tuple: (branch, trunk, targets, jacobian_ref)
    """
    print(f"Cargando dataset desde: {data_path}")
    with h5py.File(data_path, "r") as f:
        branch_inputs = f["branch_inputs"][:]
        trunk_inputs = f["trunk_inputs"][:]
        targets = f["targets"][:]

        jacobian_ref = None
        if "jacobian_reference" in f.keys():
            jacobian_ref = f["jacobian_reference"][:]

        if "params_coefficient" in f.keys():
            if branch_inputs.shape[1] < expected_branch_dim:
                coeff = f["params_coefficient"][:]
                if coeff.ndim == 1:
                    coeff = coeff[:, np.newaxis]
                branch_inputs = np.hstack([branch_inputs, coeff])

    print(f"  Branch shape : {branch_inputs.shape}")
    print(f"  Trunk shape  : {trunk_inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    if jacobian_ref is not None:
        print(f"  Jacobian ref : {jacobian_ref.shape}")
    return branch_inputs, trunk_inputs, targets, jacobian_ref


def load_model(args):
    """Instancia y carga pesos de cualquier tipo de modelo.

    Args:
        args: Argumentos parseados con model_type, branch_layers, etc.

    Returns:
        tuple: (net, scalers) con la red en eval mode y los escaladores.
    """
    import glob

    # Seleccionar clase de modelo
    if args.model_type == "jacobian":
        if args.variance_activation == "softplus":
            ModelClass = create_jacobian_deeponet_softplus(is_cartesian=True)
        else:
            ModelClass = create_jacobian_deeponet(is_cartesian=True)
        net = ModelClass(
            layer_sizes_branch=args.branch_layers,
            layer_sizes_trunk=args.trunk_layers,
            activation=args.activation,
            kernel_initializer="Glorot normal",
            jacobian_type="parameter",
            sigma=args.sigma
        )
    elif args.model_type == "vanilla_pinball":
        VanillaClass = create_vanilla_pinball_deeponet(is_cartesian=True)
        net = VanillaClass(
            layer_sizes_branch=args.branch_layers,
            layer_sizes_trunk=args.trunk_layers,
            activation=args.activation,
            kernel_initializer="Glorot normal",
            sigma=args.sigma
        )
    else:  # baseline
        net = dde.nn.DeepONetCartesianProd(
            layer_sizes_branch=args.branch_layers,
            layer_sizes_trunk=args.trunk_layers,
            activation=args.activation,
            kernel_initializer="Glorot normal"
        )

    # Buscar checkpoint
    if args.model_iter == -1:
        search1 = os.path.join(args.model_dir, f"{args.model_stem}-*.pt")
        search2 = os.path.join(args.model_dir, args.model_stem, f"{args.model_stem}-*.pt")
        files = glob.glob(search1) + glob.glob(search2)
        if not files:
            raise FileNotFoundError(
                f"No se encontraron checkpoints para {args.model_stem} en {args.model_dir}")

        def get_iter(f):
            try:
                return int(os.path.basename(f).split("-")[-1].replace(".pt", ""))
            except ValueError:
                return -1

        ckpt_path = max(files, key=get_iter)
        found_iter = get_iter(ckpt_path)
        print(f"  [Auto-detect] Usando checkpoint: {os.path.basename(ckpt_path)}")
    else:
        found_iter = args.model_iter
        ckpt_path = os.path.join(args.model_dir, f"{args.model_stem}-{found_iter}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.model_dir, args.model_stem,
                                     f"{args.model_stem}-{found_iter}.pt")

    raw_ckpt = torch.load(ckpt_path, map_location=args.device)
    state = raw_ckpt.get("model_state_dict", raw_ckpt)
    net.load_state_dict(state, strict=False)
    net.to(args.device)
    net.eval()

    # Cargar scalers
    scl_paths = [
        os.path.join(args.model_dir, f"{args.model_stem}-{found_iter}_scalers.npz"),
        os.path.join(args.model_dir, args.model_stem, f"{args.model_stem}-{found_iter}_scalers.npz"),
        os.path.join(args.model_dir, f"{args.model_stem}_scalers.npz"),
        os.path.join(args.model_dir, args.model_stem, f"{args.model_stem}_scalers.npz")
    ]
    scl_path = next((p for p in scl_paths if os.path.exists(p)), None)
    if not scl_path:
        raise FileNotFoundError(f"No se encontro el archivo de scalers para {args.model_stem}")
    scalers = dict(np.load(scl_path, allow_pickle=True))

    return net, scalers


def run_inference(net, branch_scaled, trunk_scaled, device, has_bounds=False):
    """Ejecuta inferencia sobre un batch de entradas ya escaladas.

    Args:
        net: Red PyTorch en eval mode.
        branch_scaled: Array numpy [N, D_branch] normalizado.
        trunk_scaled: Array numpy [N_nodes, D_trunk] normalizado.
        device: Dispositivo PyTorch.
        has_bounds: Si True, la red emite [y, y_lower, y_upper] concatenados.

    Returns:
        numpy array: Predicciones en espacio normalizado.
    """
    tb = torch.tensor(branch_scaled, dtype=torch.float32).to(device)
    tt = torch.tensor(trunk_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = net((tb, tt))
    return preds.cpu().numpy()


def descale_predictions(preds_scaled, scalers, has_bounds=False):
    """Desnormaliza predicciones al espacio fisico.

    Detecta automaticamente si la salida es 3x (center+lower+upper) de modelos
    Jacobian/Vanilla y maneja el broadcast correctamente.

    Args:
        preds_scaled: Predicciones en espacio normalizado.
        scalers: Diccionario con Y_mean, Y_std.
        has_bounds: Si True, retorna (center, lower, upper). Si False, solo center.

    Returns:
        tuple: (y_center, y_lower, y_upper). lower/upper son None si has_bounds=False.
    """
    n_nodes = scalers['Y_std'].shape[-1]

    # Auto-detectar salida 3x (Jacobian/Vanilla emiten center+lower+upper concatenados)
    if preds_scaled.shape[-1] == n_nodes * 3:
        y_std_ext = np.tile(scalers['Y_std'], 3)
        y_mean_ext = np.tile(scalers['Y_mean'], 3)
        preds_phys = (preds_scaled * y_std_ext) + y_mean_ext
        y_center = preds_phys[:, :n_nodes]
        y_lower = preds_phys[:, n_nodes:2*n_nodes]
        y_upper = preds_phys[:, 2*n_nodes:]
        if has_bounds:
            return y_center, y_lower, y_upper
        else:
            return y_center, None, None
    else:
        preds_phys = (preds_scaled * scalers['Y_std']) + scalers['Y_mean']
        return preds_phys, None, None


def run_montecarlo(net, branch_raw, trunk_raw, scalers, device,
                   alpha, n_mc, param_idx=None, batch_size=0):
    """Ejecuta Monte Carlo sobre todas las muestras de test.

    Para cada muestra i:
    1. Perturba la entrada N veces con ruido gaussiano (fraccion alpha).
    2. Ejecuta inferencia para cada perturbacion.
    3. Construye bandas de confianza con percentiles 5% y 95%.

    Args:
        net: Red PyTorch en eval mode.
        branch_raw: Entradas branch en espacio fisico [N_samples, D_branch].
        trunk_raw: Coordenadas trunk en espacio fisico [N_nodes, D_trunk].
        scalers: Escaladores de normalizacion.
        device: Dispositivo PyTorch.
        alpha: Fraccion de perturbacion sobre el valor base.
        n_mc: Numero de simulaciones MC por muestra.
        param_idx: Indice del parametro a perturbar (None = todo el vector).
        batch_size: Tamano de batch para inferencia. 0 = sin batching.

    Returns:
        dict: Diccionario con y_center, y_lower_mc, y_upper_mc, time_per_sample.
    """
    b_scaled = (branch_raw - scalers['branch_mean']) / scalers['branch_std']
    t_scaled = (trunk_raw - scalers['trunk_mean']) / scalers['trunk_std']
    tt = torch.tensor(t_scaled, dtype=torch.float32).to(device)

    n_samples = branch_raw.shape[0]

    # Prediccion base (sin perturbacion)
    print("\n  [MC] Prediccion base (sin ruido)...")
    base_preds_scaled = run_inference(net, b_scaled, t_scaled, device)
    y_center, _, _ = descale_predictions(base_preds_scaled, scalers)
    n_nodes = y_center.shape[1]

    # Arrays de salida
    y_lower_mc = np.zeros((n_samples, n_nodes))
    y_upper_mc = np.zeros((n_samples, n_nodes))
    mc_times = []

    print(f"  [MC] Ejecutando {n_mc} simulaciones x {n_samples} muestras (alpha={alpha})...")
    for i in range(n_samples):
        t0 = time.time()

        val_base = branch_raw[i]  # [D_branch]

        # Generar perturbaciones
        noise = np.random.normal(0, 1, (n_mc, 1))

        if param_idx is not None:
            # Perturbar solo un parametro (ej. viscosidad)
            branch_pert_raw = np.tile(val_base, (n_mc, 1))
            gen_base = val_base[param_idx]
            branch_pert_raw[:, param_idx] = gen_base * (1 + alpha * noise.flatten())
        else:
            # Perturbar todo el vector de entrada
            branch_pert_raw = val_base * (1 + alpha * noise)

        branch_pert_scaled = (branch_pert_raw - scalers['branch_mean']) / scalers['branch_std']

        # Inferencia de las N perturbaciones
        if batch_size > 0:
            preds_list = []
            with torch.no_grad():
                for j in range(0, n_mc, batch_size):
                    tb = torch.tensor(branch_pert_scaled[j:j+batch_size],
                                      dtype=torch.float32).to(device)
                    p = net((tb, tt))
                    preds_list.append(p.cpu())
            preds_scaled = torch.cat(preds_list, dim=0).numpy()
        else:
            preds_scaled = run_inference(net, branch_pert_scaled, t_scaled, device)

        # Desnormalizar
        preds_phys, _, _ = descale_predictions(preds_scaled, scalers)

        # Extraer percentiles para bandas de confianza
        y_lower_mc[i] = np.percentile(preds_phys, 5, axis=0).flatten()[:n_nodes]
        y_upper_mc[i] = np.percentile(preds_phys, 95, axis=0).flatten()[:n_nodes]

        mc_times.append(time.time() - t0)

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"    Muestra {i+1}/{n_samples} "
                  f"(t_avg={np.mean(mc_times):.3f}s/muestra)")

    avg_time = np.mean(mc_times)
    print(f"  [MC] Completado. Tiempo promedio: {avg_time:.3f}s/muestra")

    return {
        "y_center": y_center,
        "y_lower_mc": y_lower_mc,
        "y_upper_mc": y_upper_mc,
        "time_per_sample": avg_time
    }


def compute_metrics(targets, y_center, y_lower, y_upper, jac_ref=None):
    """Calcula MSE, cobertura, MPIW y correlaciones topologicas.

    Args:
        targets: Ground truth [N_samples, N_nodes].
        y_center: Prediccion central [N_samples, N_nodes].
        y_lower: Banda inferior [N_samples, N_nodes].
        y_upper: Banda superior [N_samples, N_nodes].
        jac_ref: Referencia de sensibilidad del HDF5 (opcional).

    Returns:
        dict: Metricas calculadas.
    """
    metrics = {}

    # MSE central
    metrics["mse"] = float(np.mean((targets - y_center) ** 2))
    metrics["nrmse"] = float(np.linalg.norm(targets - y_center) / np.linalg.norm(targets)) * 100

    # Cobertura empirica
    mask = (targets >= y_lower) & (targets <= y_upper)
    metrics["coverage"] = float(np.mean(mask) * 100)

    # MPIW
    widths = y_upper - y_lower
    metrics["mpiw"] = float(np.mean(widths))

    # NLL computation
    std_approx = widths / (2 * 1.644853)
    var_approx = std_approx**2 + 1e-8
    metrics["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * var_approx) + 0.5 * ((targets - y_center)**2) / var_approx))

    # Correlacion ancho vs |error|
    flat_widths = widths.flatten()
    flat_error = np.abs((targets - y_center).flatten())
    try:
        metrics["corr_err_pearson"], _ = pearsonr(flat_widths, flat_error)
        metrics["corr_err_spearman"], _ = spearmanr(flat_widths, flat_error)
    except Exception:
        metrics["corr_err_pearson"] = None
        metrics["corr_err_spearman"] = None

    # Correlacion ancho vs |sensibilidad| (si existe referencia)
    metrics["corr_sens_pearson"] = None
    metrics["corr_sens_spearman"] = None
    if jac_ref is not None:
        if jac_ref.ndim == 3:
            jac_ref_processed = np.sum(jac_ref, axis=2)
        else:
            jac_ref_processed = jac_ref
        flat_sens = np.abs(jac_ref_processed.flatten())
        try:
            metrics["corr_sens_pearson"], _ = pearsonr(flat_widths, flat_sens)
            metrics["corr_sens_spearman"], _ = spearmanr(flat_widths, flat_sens)
        except Exception:
            pass

    # Cobertura y MPIW por Decil de Error
    metrics["deciles"] = compute_decile_metrics(targets, y_center, y_lower, y_upper, n_deciles=10)

    return metrics


def compute_decile_metrics(targets, y_center, y_lower, y_upper, n_deciles=10):
    """Calcula cobertura, MPIW, NLL y MSE por decil de error (ordenado por MSE ascendente)."""
    mse_per_sample = np.mean((targets - y_center) ** 2, axis=1)
    cov_per_sample = np.mean((targets >= y_lower) & (targets <= y_upper), axis=1) * 100.0
    mpiw_per_sample = np.mean(y_upper - y_lower, axis=1)
    
    std_approx = (y_upper - y_lower) / (2 * 1.644853)
    var_approx = std_approx**2 + 1e-8
    nll_per_sample = np.mean(0.5 * np.log(2 * np.pi * var_approx) + 0.5 * ((targets - y_center)**2) / var_approx, axis=1)

    sorted_idx = np.argsort(mse_per_sample)
    sorted_mse = mse_per_sample[sorted_idx]
    sorted_cov = cov_per_sample[sorted_idx]
    sorted_mpiw = mpiw_per_sample[sorted_idx]
    sorted_nll = nll_per_sample[sorted_idx]

    decile_mse = np.array_split(sorted_mse, n_deciles)
    decile_cov = np.array_split(sorted_cov, n_deciles)
    decile_mpiw = np.array_split(sorted_mpiw, n_deciles)
    decile_nll = np.array_split(sorted_nll, n_deciles)

    return {
        "decile_idx": list(range(1, n_deciles + 1)),
        "coverage_mean": [float(np.mean(d)) for d in decile_cov],
        "coverage_std": [float(np.std(d)) for d in decile_cov],
        "mpiw_mean": [float(np.mean(d)) for d in decile_mpiw],
        "nll_mean": [float(np.mean(d)) for d in decile_nll],
        "mse_mean": [float(np.mean(d)) for d in decile_mse],
        "mse_range": [(float(np.min(d)), float(np.max(d))) for d in decile_mse],
        "n_samples": [len(d) for d in decile_mse],
    }


def print_report(metrics, mc_time, n_mc, alpha):
    """Imprime el reporte en formato consola, incluyendo metricas globales y por decil."""
    sep = "=" * 75
    print(f"\n{sep}")
    print("  MONTE CARLO BENCHMARK EVALUATOR")
    print(f"  Simulaciones: {n_mc} | Alpha: {alpha}")
    print(sep)
    print(f"  MSE Central              : {metrics['mse']:.4e}  (nRMSE: {metrics['nrmse']:.2f}%)")
    print(f"  Negative Log-Likelihood  : {metrics.get('nll', 0.0):.4f}")
    print("-" * 75)
    print("  [Metricas Globales de Incertidumbre - Monte Carlo]")
    print(f"  Cobertura Empirica (PICP): {metrics['coverage']:.2f} %")
    print(f"  MPIW (Ancho Promedio)    : {metrics['mpiw']:.4e}")
    if metrics.get("corr_sens_pearson") is not None:
        print(f"  Correlacion Ancho vs |Sens|: Pearson = {metrics['corr_sens_pearson']:.3f}"
              f" | Spearman = {metrics['corr_sens_spearman']:.3f}")
    if metrics.get("corr_err_pearson") is not None:
        print(f"  Correlacion Ancho vs |Err| : Pearson = {metrics['corr_err_pearson']:.3f}"
              f" | Spearman = {metrics['corr_err_spearman']:.3f}")
    
    if "deciles" in metrics:
        dec = metrics["deciles"]
        print("-" * 75)
        print("  [Cobertura, MPIW y NLL Promedio por Decil de Error]")
        print(f"  (Decil 1 = Errores mas bajos | Decil {len(dec['decile_idx'])} = Errores mas altos)")
        print(f"  {'Decil':<8} | {'MSE Rango':<26} | {'Cobertura':<14} | {'MPIW':<12} | {'NLL':<10} | {'N':<5}")
        print("  " + "-" * 82)
        for i, d in enumerate(dec["decile_idx"]):
            lo, hi = dec["mse_range"][i]
            cov = dec["coverage_mean"][i]
            mpiw = dec["mpiw_mean"][i]
            nll_d = dec["nll_mean"][i]
            n = dec["n_samples"][i]
            print(f"  Decil {d:<2} | [{lo:.2e}, {hi:.2e}] | {cov:6.2f} %       | {mpiw:.4e}  | {nll_d:8.4f}   | {n}")

    print("-" * 75)
    print(f"  Tiempo promedio MC/muestra: {mc_time:.4f} s  ({n_mc} sims)")
    print(sep + "\n")


def main():
    args = parse_args()

    dde.config.set_default_float("float32")

    # Cargar datos
    branch, trunk, targets, jac_ref = load_data(args.data_path, args.branch_layers[0])

    # Cargar modelo
    print(f"\n[Cargando] Modelo tipo={args.model_type}: {args.model_stem}")
    net, scalers = load_model(args)

    # Ejecutar Monte Carlo
    mc_results = run_montecarlo(
        net, branch, trunk, scalers, args.device,
        alpha=args.alpha,
        n_mc=args.n_mc,
        param_idx=args.param_idx,
        batch_size=args.infer_batch_size
    )

    # Calcular metricas
    metrics = compute_metrics(
        targets,
        mc_results["y_center"],
        mc_results["y_lower_mc"],
        mc_results["y_upper_mc"],
        jac_ref
    )

    # Reporte
    print_report(metrics, mc_results["time_per_sample"], args.n_mc, args.alpha)


if __name__ == "__main__":
    main()
