"""
abstract_benchmark.py

Script unificado y abstracto para evaluar el desempeño (Forward Pass) y la 
cuantificación de incertidumbre (UQ) de Jacobian-DeepONet vs. DeepONet Clásica 
para cualquier ecuación del proyecto (Burgers, Biharmónica, Darcy, NS2D).

Este script NO utiliza Monte Carlo. Solo computa las predicciones deterministas
de ambos modelos y evalúa directamente las bandas asimétricas nativas del modelo Jacobiano.

Métricas entregadas:
- MSE Predicción Central (Baseline y Jacobiano)
- Cobertura Empírica de Incertidumbre (Jacobiano)
- MPIW (Mean Prediction Interval Width)
- Correlación (Pearson/Spearman) entre el ancho de banda y la magnitud de la Sensibilidad Física (Ground Truth).
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import h5py
from scipy.stats import pearsonr, spearmanr

# ── Configuración de Paths ──────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [REPO_ROOT,
          os.path.join(REPO_ROOT, "deepxde-extensions"),
          os.path.join(REPO_ROOT, "benchmarks")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import deepxde as dde
from jacobian_deeponet import create_jacobian_deeponet
from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus

def parse_args():
    """Analiza y consolida los argumentos CLI paramétricos abstractos."""
    parser = argparse.ArgumentParser(description="Abstract Benchmark Evaluator (No Monte Carlo)")
    
    # Rutas
    parser.add_argument("--data_path", type=str, required=True,
                        help="Ruta absoluta o relativa al dataset de prueba (.h5).")
    parser.add_argument("--baseline_dir", type=str, default="",
                        help="Directorio del modelo Baseline clásico (.pt y scalers).")
    parser.add_argument("--baseline_stem", type=str, default="",
                        help="Nombre base del modelo clásico (ej. burgers_C_DeepThin).")
    parser.add_argument("--baseline_iter", type=int, default=-1,
                        help="Iteración del checkpoint Baseline a cargar. Si es -1, buscará automáticamente el último .pt")
    
    parser.add_argument("--jacobian_dir", type=str, default="",
                        help="Directorio del modelo Jacobian-DeepONet (.pt y scalers).")
    parser.add_argument("--jacobian_stem", type=str, default="",
                        help="Nombre base del modelo jacobiano (ej. burgers_Jac_A_DeepThin).")
    parser.add_argument("--jacobian_iter", type=int, default=-1,
                        help="Iteración del checkpoint Jacobian a cargar.")
    
    parser.add_argument("--vanilla_dir", type=str, default="",
                        help="Directorio del modelo Vanilla Pinball.")
    parser.add_argument("--vanilla_stem", type=str, default="",
                        help="Nombre base del modelo Vanilla Pinball.")
    parser.add_argument("--vanilla_iter", type=int, default=-1,
                        help="Iteración del checkpoint Vanilla a cargar.")
    
    # Arquitectura
    parser.add_argument("--branch_layers", type=int, nargs="+", required=True,
                        help="Arquitectura de la Branch (ej. 4098 256 256 256).")
    parser.add_argument("--trunk_layers", type=int, nargs="+", required=True,
                        help="Arquitectura de la Trunk (ej. 2 256 256 256).")
    parser.add_argument("--activation", type=str, default="swish",
                        help="Función de activación general.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Sigma escalar utilizado en el Jacobiano.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositivo PyTorch (cpu/cuda).")
    parser.add_argument("--variance_activation", type=str, choices=["exp", "softplus"], default="exp",
                        help="Función de activación para la varianza en JacobianDeepONet.")
    parser.add_argument("--infer_batch_size", type=int, default=0,
                        help="Tamaño de lote para inferencia. Si es 0, no usa batches.")
                        
    return parser.parse_args()


def load_smart_data(data_path: str, expected_branch_dim: int):
    """
    Lee de forma robusta los tensores HDF5 adaptándose a las diferentes estructuras
    de las ecuaciones del proyecto. Si existen parámetros anexos, los concatena
    solo si el modelo los espera en su dimensión de entrada.

    Args:
        data_path (str): Ruta al .h5 de testing.
        expected_branch_dim (int): Dimensión esperada para la rama (args.branch_layers[0]).

    Returns:
        tuple: (branch, trunk, targets, jacobian_ref)
    """
    print(f"Cargando dataset desde: {data_path}")
    with h5py.File(data_path, "r") as f:
        branch_inputs = f["branch_inputs"][:]
        trunk_inputs  = f["trunk_inputs"][:]
        targets       = f["targets"][:]
        
        # Extracción opcional del Ground Truth de la sensibilidad
        if "jacobian_reference" in f.keys():
            jacobian_ref = f["jacobian_reference"][:]
        else:
            jacobian_ref = None
            print("  Advertencia: 'jacobian_reference' no encontrado en HDF5.")
            
        # Concatenación inteligente de coeficientes anexos (Burgers/NS2D)
        if "params_coefficient" in f.keys():
            if branch_inputs.shape[1] < expected_branch_dim:
                coeff = f["params_coefficient"][:]
                if coeff.ndim == 1:
                    coeff = coeff[:, np.newaxis]
                branch_inputs = np.hstack([branch_inputs, coeff])
            
    print(f"  Branch shape : {branch_inputs.shape}")
    print(f"  Trunk shape  : {trunk_inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    return branch_inputs, trunk_inputs, targets, jacobian_ref


def load_pytorch_model(model_class, branch_layers, trunk_layers, act, 
                       model_dir, stem, iter_num, device, is_jacobian=False, sigma=0.05):
    """
    Instancia una red y restaura sus pesos dict_state desde disco de forma flexible.
    """
    if is_jacobian:
        net = model_class(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation=act,
            kernel_initializer="Glorot normal",
            jacobian_type="parameter",
            sigma=sigma
        )
    elif "Vanilla" in model_class.__name__:
        net = model_class(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation=act,
            kernel_initializer="Glorot normal",
            sigma=sigma
        )
    else:
        net = model_class(
            layer_sizes_branch=branch_layers,
            layer_sizes_trunk=trunk_layers,
            activation=act,
            kernel_initializer="Glorot normal"
        )
        
    if iter_num == -1:
        import glob
        search_path1 = os.path.join(model_dir, f"{stem}-*.pt")
        search_path2 = os.path.join(model_dir, stem, f"{stem}-*.pt")
        files = glob.glob(search_path1) + glob.glob(search_path2)
        if not files:
            raise FileNotFoundError(f"No se encontraron checkpoints automáticamente para {stem} en {model_dir}")
        def get_iter(f):
            try: return int(os.path.basename(f).split("-")[-1].replace(".pt", ""))
            except: return -1
        ckpt_path = max(files, key=get_iter)
        found_iter = get_iter(ckpt_path)
        print(f"    [Auto-detect] Usando checkpoint: {os.path.basename(ckpt_path)}")
    else:
        found_iter = iter_num
        ckpt_path = os.path.join(model_dir, f"{stem}-{iter_num}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(model_dir, stem, f"{stem}-{iter_num}.pt")
        
    try:
        raw_ckpt = torch.load(ckpt_path, map_location=device)
        state = raw_ckpt.get("model_state_dict", raw_ckpt)
        net.load_state_dict(state, strict=False)
    except Exception as e:
        raise FileNotFoundError(f"Error cargando pesos desde {ckpt_path}: {e}")
        
    net.to(device)
    net.eval()
    
    # Cargar scalers (intenta con y sin iteración)
    scl_paths = [
        os.path.join(model_dir, f"{stem}-{found_iter}_scalers.npz"),
        os.path.join(model_dir, stem, f"{stem}-{found_iter}_scalers.npz"),
        os.path.join(model_dir, f"{stem}_scalers.npz"),
        os.path.join(model_dir, stem, f"{stem}_scalers.npz")
    ]
    scl_path = next((p for p in scl_paths if os.path.exists(p)), None)
    if not scl_path:
        raise FileNotFoundError(f"No se encontró el archivo de scalers para {stem} en {model_dir}")
    scalers = dict(np.load(scl_path, allow_pickle=True))
    
    return net, scalers


def forward_pass_inference(net, branch, trunk, scalers, device, has_bounds=False, batch_size=0):
    """
    Ejecuta un pase de inferencia sobre toda la malla de prueba procesado por lotes (chunks)
    para evitar errores de Out of Memory (OOM) al computar jacobianos masivos.

    Returns:
        tuple: (y_center, y_lower, y_upper) desescalados a dominio físico.
               y_lower e y_upper serán None si has_bounds=False.
    """
    b_scaled = (branch - scalers['branch_mean']) / scalers['branch_std']
    t_scaled = (trunk - scalers['trunk_mean']) / scalers['trunk_std']
    
    # El trunk suele ser fijo para todas las simulaciones en estos datasets cartesianos
    tt = torch.tensor(t_scaled, dtype=torch.float32).to(device)
    
    N = b_scaled.shape[0]
    
    if batch_size is None or batch_size <= 0:
        tb = torch.tensor(b_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_scaled = net((tb, tt))
        preds_np = preds_scaled.cpu().numpy()
    else:
        preds_list = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                tb = torch.tensor(b_scaled[i:i+batch_size], dtype=torch.float32).to(device)
                preds_scaled = net((tb, tt))
                preds_list.append(preds_scaled.cpu())
                
        preds_np = torch.cat(preds_list, dim=0).numpy()
    
    # Des-escalamiento al espacio físico
    if has_bounds and preds_np.shape[-1] == scalers['Y_std'].shape[-1] * 3:
        y_std_ext = np.tile(scalers['Y_std'], 3)
        y_mean_ext = np.tile(scalers['Y_mean'], 3)
        preds_phys = (preds_np * y_std_ext) + y_mean_ext
        
        n_n = preds_phys.shape[-1] // 3
        y_center = preds_phys[:, 0:n_n]
        y_lower = preds_phys[:, n_n:2*n_n]
        y_upper = preds_phys[:, 2*n_n:]
    else:
        preds_phys = (preds_np * scalers['Y_std']) + scalers['Y_mean']
        y_center = preds_phys
        y_lower = None
        y_upper = None
        
    return y_center, y_lower, y_upper


def compute_uq_metrics(targets, center, lower, upper, jac_ref):
    """Calcula metricas UQ para un modelo dado."""
    metrics = {}
    if center is None:
        return metrics
        
    metrics["mse"] = float(np.mean((targets - center) ** 2))
    metrics["nrmse"] = float(np.linalg.norm(targets - center) / np.linalg.norm(targets)) * 100
    
    mask = (targets >= lower) & (targets <= upper)
    metrics["coverage"] = float(np.mean(mask) * 100)
    
    widths = upper - lower
    metrics["mpiw"] = float(np.mean(widths))
    
    if jac_ref is not None:
        if jac_ref.ndim == 3:
            jac_ref_processed = np.sum(jac_ref, axis=2)
        else:
            jac_ref_processed = jac_ref
            
        flat_widths = widths.flatten()
        flat_sens = np.abs(jac_ref_processed.flatten())
        metrics["corr_pearson"], _ = pearsonr(flat_widths, flat_sens)
        metrics["corr_spearman"], _ = spearmanr(flat_widths, flat_sens)
        
    flat_widths_all = widths.flatten()
    flat_error = np.abs((targets - center).flatten())
    metrics["corr_err_pearson"], _ = pearsonr(flat_widths_all, flat_error)
    metrics["corr_err_spearman"], _ = spearmanr(flat_widths_all, flat_error)
    
    return metrics

def compute_metrics(targets, base_center, jac_center, jac_lower, jac_upper, van_center, van_lower, van_upper, jac_ref):
    """
    Calcula los descriptores estadísticos, cobertura, MPIW y correlaciones topológicas.
    """
    metrics = {
        "base": {},
        "jac": compute_uq_metrics(targets, jac_center, jac_lower, jac_upper, jac_ref),
        "van": compute_uq_metrics(targets, van_center, van_lower, van_upper, jac_ref)
    }
    
    if base_center is not None:
        metrics["base"]["mse"] = float(np.mean((targets - base_center) ** 2))
        metrics["base"]["nrmse"] = float(np.linalg.norm(targets - base_center) / np.linalg.norm(targets)) * 100
        
    return metrics


def print_report(metrics):
    """Dibuja el resumen en formato abstracto por consola."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  ABSTRACT BENCHMARK EVALUATOR (No Monte Carlo)")
    print(sep)
    if "mse" in metrics["base"]:
        print(f"  MSE Central (Baseline)         : {metrics['base']['mse']:.4e}  (nRMSE: {metrics['base']['nrmse']:.2f}%)")
    
    if "mse" in metrics["jac"]:
        m = metrics["jac"]
        print("-" * 70)
        print("  [Jacobian-DeepONet]")
        print(f"  MSE Central                    : {m['mse']:.4e}  (nRMSE: {m['nrmse']:.2f}%)")
        print(f"  Cobertura Empírica             : {m['coverage']:.2f} %")
        print(f"  MPIW (Ancho Promedio)          : {m['mpiw']:.4e}")
        if m.get("corr_pearson") is not None:
            print(f"  Correlación Ancho vs |Sens|    : Pearson = {m['corr_pearson']:.3f} | Spearman = {m['corr_spearman']:.3f}")
        if m.get("corr_err_pearson") is not None:
            print(f"  Correlación Ancho vs |Err|     : Pearson = {m['corr_err_pearson']:.3f} | Spearman = {m['corr_err_spearman']:.3f}")
            
    if "mse" in metrics["van"]:
        m = metrics["van"]
        print("-" * 70)
        print("  [Vanilla Pinball - Ablación]")
        print(f"  MSE Central                    : {m['mse']:.4e}  (nRMSE: {m['nrmse']:.2f}%)")
        print(f"  Cobertura Empírica             : {m['coverage']:.2f} %")
        print(f"  MPIW (Ancho Promedio)          : {m['mpiw']:.4e}")
        if m.get("corr_pearson") is not None:
            print(f"  Correlación Ancho vs |Sens|    : Pearson = {m['corr_pearson']:.3f} | Spearman = {m['corr_spearman']:.3f}")
        if m.get("corr_err_pearson") is not None:
            print(f"  Correlación Ancho vs |Err|     : Pearson = {m['corr_err_pearson']:.3f} | Spearman = {m['corr_err_spearman']:.3f}")
            
    print(sep + "\n")


def main():
    args = parse_args()
    
    branch, trunk, targets, jac_ref = load_smart_data(args.data_path, args.branch_layers[0])
    
    base_center = None
    if args.baseline_stem:
        # ── Línea Base Clásica ──
        print("\n[Inferencia] Modelo Baseline Clásico...")
        net_base, scalers_base = load_pytorch_model(
            model_class=dde.nn.DeepONetCartesianProd,
            branch_layers=args.branch_layers,
            trunk_layers=args.trunk_layers,
            act=args.activation,
            model_dir=args.baseline_dir,
            stem=args.baseline_stem,
            iter_num=args.baseline_iter,
            device=args.device,
            is_jacobian=False
        )
        
        t0 = time.time()
        base_center, _, _ = forward_pass_inference(net_base, branch, trunk, scalers_base, args.device, has_bounds=False, batch_size=args.infer_batch_size)
        print(f"  Forward pass completado en {time.time() - t0:.2f} s")
    
    jac_center, jac_lower, jac_upper = None, None, None
    if args.jacobian_stem:
        # ── Jacobian-DeepONet ──
        print("\n[Inferencia] Modelo Jacobian-DeepONet...")
        if args.variance_activation == "softplus":
            print("    [Info] Usando JacobianDeepONetSoftplus (Legacy)")
            JacobianDeepONet = create_jacobian_deeponet_softplus(is_cartesian=True)
        else:
            print("    [Info] Usando JacobianDeepONet (Exp + Normalización)")
            JacobianDeepONet = create_jacobian_deeponet(is_cartesian=True)
            
        net_jac, scalers_jac = load_pytorch_model(
            model_class=JacobianDeepONet,
            branch_layers=args.branch_layers,
            trunk_layers=args.trunk_layers,
            act=args.activation,
            model_dir=args.jacobian_dir,
            stem=args.jacobian_stem,
            iter_num=args.jacobian_iter,
            device=args.device,
            is_jacobian=True,
            sigma=args.sigma
        )
        
        t0 = time.time()
        jac_center, jac_lower, jac_upper = forward_pass_inference(net_jac, branch, trunk, scalers_jac, args.device, has_bounds=True, batch_size=args.infer_batch_size)
        print(f"  Forward pass completado en {time.time() - t0:.2f} s")
        
    van_center, van_lower, van_upper = None, None, None
    if args.vanilla_stem:
        # ── Vanilla Pinball DeepONet ──
        print("\n[Inferencia] Modelo Vanilla Pinball DeepONet...")
        from vanilla_pinball_deeponet import create_vanilla_pinball_deeponet
        VanillaPinballClass = create_vanilla_pinball_deeponet(is_cartesian=True)
            
        net_van, scalers_van = load_pytorch_model(
            model_class=VanillaPinballClass,
            branch_layers=args.branch_layers,
            trunk_layers=args.trunk_layers,
            act=args.activation,
            model_dir=args.vanilla_dir,
            stem=args.vanilla_stem,
            iter_num=args.vanilla_iter,
            device=args.device,
            is_jacobian=False,
            sigma=args.sigma
        )
        
        t0 = time.time()
        van_center, van_lower, van_upper = forward_pass_inference(net_van, branch, trunk, scalers_van, args.device, has_bounds=True, batch_size=args.infer_batch_size)
        print(f"  Forward pass completado en {time.time() - t0:.2f} s")
    
    # ── Métricas y Reporte ──
    metrics = compute_metrics(targets, base_center, jac_center, jac_lower, jac_upper, van_center, van_lower, van_upper, jac_ref)
    print_report(metrics)


if __name__ == "__main__":
    main()
