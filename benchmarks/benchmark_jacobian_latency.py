"""
benchmark_jacobian_latency.py

Modulo para medir de forma estandarizada y rigurosa la latencia y throughput
de inferencia de Jacobian-DeepONet, Baseline DeepONet y Vanilla Pinball DeepONet.

Adopta la misma estructura de argumentos y carga de datos/modelos que
abstract_benchmark.py, garantizando compatibilidad total y mediciones precisas
para comparacion directa contra metodos estocasticos como Monte Carlo (MC).

Metricas calculadas:
- Latencia promedio por muestra (ms/sample y s/sample)
- Desviacion estandar de la latencia por muestra (ms)
- Throughput promedio (samples/sec)
- Tiempo total promedio por pasada sobre el conjunto de evaluacion (s)
- Factores de aceleracion relativa (Speedup)
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import h5py

# Configuracion de Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [REPO_ROOT,
          os.path.join(REPO_ROOT, "deepxde-extensions"),
          os.path.join(REPO_ROOT, "benchmarks")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import deepxde as dde
from abstract_benchmark import load_smart_data, load_pytorch_model
from jacobian_deeponet import create_jacobian_deeponet
from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus
from vanilla_pinball_deeponet import create_vanilla_pinball_deeponet


def parse_args():
    """
    Analiza los argumentos de linea de comandos manteniendo compatibilidad
    con el formato de abstract_benchmark.py.

    Returns:
        argparse.Namespace: Objeto con los parametros de configuracion e inferencia.
    """
    parser = argparse.ArgumentParser(
        description="Unified Latency and Throughput Benchmarking for DeepONet Architectures"
    )

    # Rutas y descripcion
    parser.add_argument("--data_path", type=str, required=True,
                        help="Ruta al dataset de prueba en formato HDF5 (.h5).")
    parser.add_argument("--equation_name", type=str, default="PDE Benchmark",
                        help="Nombre descriptivo de la ecuacion diferencial evaluada.")

    # Modelos: Baseline Clásico
    parser.add_argument("--baseline_dir", type=str, default="",
                        help="Directorio del modelo Baseline (.pt y scalers).")
    parser.add_argument("--baseline_stem", type=str, default="",
                        help="Nombre base del checkpoint del modelo Baseline.")
    parser.add_argument("--baseline_iter", type=int, default=-1,
                        help="Numero de iteracion del checkpoint Baseline (-1 para autodetectar).")

    # Modelos: Jacobian-DeepONet
    parser.add_argument("--jacobian_dir", type=str, default="",
                        help="Directorio del modelo Jacobian-DeepONet (.pt y scalers).")
    parser.add_argument("--jacobian_stem", type=str, default="",
                        help="Nombre base del checkpoint del modelo Jacobian-DeepONet.")
    parser.add_argument("--jacobian_iter", type=int, default=-1,
                        help="Numero de iteracion del checkpoint Jacobian (-1 para autodetectar).")

    # Modelos: Vanilla Pinball (Ablacion)
    parser.add_argument("--vanilla_dir", type=str, default="",
                        help="Directorio del modelo Vanilla Pinball (.pt y scalers).")
    parser.add_argument("--vanilla_stem", type=str, default="",
                        help="Nombre base del checkpoint del modelo Vanilla Pinball.")
    parser.add_argument("--vanilla_iter", type=int, default=-1,
                        help="Numero de iteracion del checkpoint Vanilla (-1 para autodetectar).")

    # Arquitectura de red
    parser.add_argument("--branch_layers", type=int, nargs="+", required=True,
                        help="Topologia de capas de la Branch Network (ej. 129 256 256 256).")
    parser.add_argument("--trunk_layers", type=int, nargs="+", required=True,
                        help="Topologia de capas de la Trunk Network (ej. 2 256 256 256).")
    parser.add_argument("--activation", type=str, default="swish",
                        help="Funcion de activacion no lineal en capas ocultas.")
    parser.add_argument("--sigma", type=float, default=0.025,
                        help="Escala de perturbacion sigma para incertidumbre analitica.")
    parser.add_argument("--variance_activation", type=str, choices=["exp", "softplus"],
                        default="softplus", help="Activacion para multiplicadores de varianza.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositivo de computo para tensores PyTorch (cpu/cuda).")
    parser.add_argument("--infer_batch_size", type=int, default=0,
                        help="Tamano de batch para inferencia (0 para procesar todo el conjunto en un lote).")

    # Parametros especificos de medicion de latencia
    parser.add_argument("--n_warmup", type=int, default=10,
                        help="Numero de pasadas de calentamiento iniciales sin medir.")
    parser.add_argument("--n_repeats", type=int, default=50,
                        help="Numero de repeticiones cronometradas para el analisis estadistico.")

    return parser.parse_args()


def benchmark_model_latency(net, branch, trunk, scalers, device, batch_size=0,
                            n_warmup=10, n_repeats=50):
    """
    Ejecuta el protocolo de medicion cronometrada de latencia y throughput
    sobre un modelo DeepONet dado, aplicando calentamiento previo y sincronizacion CUDA.

    Args:
        net (torch.nn.Module): Red neuronal en modo de evaluacion (eval).
        branch (np.ndarray): Tensor de entradas para la Branch Network.
        trunk (np.ndarray): Tensor de coordenadas espaciales para la Trunk Network.
        scalers (dict): Diccionario con medias y desviaciones estandar de normalizacion.
        device (str): Dispositivo destino ('cuda' o 'cpu').
        batch_size (int): Tamano de lote (0 = inferencia en un solo tensor).
        n_warmup (int): Cantidad de iteraciones de warmup no cronometradas.
        n_repeats (int): Cantidad de iteraciones cronometradas para calcular estadisticas.

    Returns:
        dict: Metricas de latencia por muestra, desviacion estandar, throughput y tiempo total.
    """
    b_scaled = (branch - scalers['branch_mean']) / scalers['branch_std']
    t_scaled = (trunk - scalers['trunk_mean']) / scalers['trunk_std']

    tt = torch.tensor(t_scaled, dtype=torch.float32).to(device)
    tb_all = torch.tensor(b_scaled, dtype=torch.float32).to(device)

    num_samples = b_scaled.shape[0]
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # 1. Warm-up (Estabilizacion de memoria y caches de ejecucion)
    with torch.no_grad():
        for _ in range(n_warmup):
            if batch_size is None or batch_size <= 0:
                _ = net((tb_all, tt))
            else:
                for i in range(0, num_samples, batch_size):
                    tb = tb_all[i:i + batch_size]
                    _ = net((tb, tt))
            if is_cuda:
                torch.cuda.synchronize()

    # 2. Medicion estadística cronometrada
    latencies_sec = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if batch_size is None or batch_size <= 0:
                _ = net((tb_all, tt))
            else:
                for i in range(0, num_samples, batch_size):
                    tb = tb_all[i:i + batch_size]
                    _ = net((tb, tt))

            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_sec.append(t1 - t0)

    latencies_sec = np.array(latencies_sec)
    per_sample_ms = (latencies_sec * 1000.0) / num_samples
    per_sample_s = latencies_sec / num_samples
    throughput = num_samples / latencies_sec

    return {
        "num_samples": num_samples,
        "batch_size": batch_size if (batch_size and batch_size > 0) else num_samples,
        "total_time_avg_s": float(np.mean(latencies_sec)),
        "total_time_std_s": float(np.std(latencies_sec)),
        "per_sample_avg_ms": float(np.mean(per_sample_ms)),
        "per_sample_std_ms": float(np.std(per_sample_ms)),
        "per_sample_avg_s": float(np.mean(per_sample_s)),
        "throughput_avg": float(np.mean(throughput)),
        "throughput_std": float(np.std(throughput))
    }


def print_latency_report(equation_name, device, dataset_info, benchmark_results):
    """
    Imprime un informe consolidado en consola con el formato estandarizado
    del proyecto para facilitar comparaciones directas contra Monte Carlo.

    Args:
        equation_name (str): Titulo descriptivo de la ecuacion diferencial.
        device (str): Dispositivo utilizado ('cuda' o 'cpu').
        dataset_info (dict): Dimensiones y estructura del dataset evaluado.
        benchmark_results (dict): Resultados de latencia obtenidos para cada modelo.
    """
    sep = "=" * 80
    subsep = "-" * 80

    print(f"\n{sep}")
    print(f"  BENCHMARK DE LATENCIA Y THROUGHPUT DE INFERENCIA")
    print(f"  Ecuacion: {equation_name} | Dispositivo: {device}")
    print(f"  Muestras: {dataset_info['N']} | Sensores Branch: {dataset_info['branch_dim']} | Nodos Trunk: {dataset_info['trunk_nodes']}")
    print(sep)

    print(f"\n{'Modelo':<26} | {'Latencia (ms/muestra)':<25} | {'Latencia (s/muestra)':<20} | {'Throughput (samples/s)':<22}")
    print(subsep)

    base_lat = None
    if "Baseline" in benchmark_results:
        base_lat = benchmark_results["Baseline"]["per_sample_avg_ms"]

    for name, r in benchmark_results.items():
        lat_ms_str = f"{r['per_sample_avg_ms']:8.4f} +/- {r['per_sample_std_ms']:6.4f}"
        lat_s_str = f"{r['per_sample_avg_s']:.6f}"
        thr_str = f"{r['throughput_avg']:8.1f} +/- {r['throughput_std']:5.1f}"
        print(f"{name:<26} | {lat_ms_str:<25} | {lat_s_str:<20} | {thr_str:<22}")

    print(subsep)

    # Detalle de comparacion de latencia para UQ
    if "Jacobian-DeepONet" in benchmark_results:
        jac_res = benchmark_results["Jacobian-DeepONet"]
        print(f"\n  [Resumen para Comparacion UQ vs Monte Carlo]")
        print(f"  - Tiempo inferencia UQ (Jacobian-DeepONet): {jac_res['per_sample_avg_s']:.6f} s/muestra ({jac_res['per_sample_avg_ms']:.4f} ms)")
        print(f"  - Throughput UQ (Jacobian-DeepONet)      : {jac_res['throughput_avg']:.1f} muestras/s")

        if base_lat is not None and base_lat > 0:
            overhead = jac_res["per_sample_avg_ms"] / base_lat
            print(f"  - Overhead relativo vs Baseline clasico  : {overhead:.2f}x (Forward pass UQ completo vs solo Media)")

    print(f"{sep}\n")


def main():
    """Funcion principal de ejecucion del benchmark de latencia."""
    args = parse_args()
    dde.config.set_default_float("float32")

    # Carga de datos con verificacion automatica de dimensiones y coeficientes anexos
    branch, trunk, targets, jac_ref = load_smart_data(args.data_path, args.branch_layers[0])

    dataset_info = {
        "N": branch.shape[0],
        "branch_dim": branch.shape[1],
        "trunk_nodes": trunk.shape[0]
    }

    benchmark_results = {}

    # 1. Baseline DeepONet
    if args.baseline_stem:
        print(f"\n[Cargando Modelo] Baseline Clasico: {args.baseline_stem}")
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
        res_base = benchmark_model_latency(
            net=net_base,
            branch=branch,
            trunk=trunk,
            scalers=scalers_base,
            device=args.device,
            batch_size=args.infer_batch_size,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats
        )
        benchmark_results["Baseline"] = res_base

    # 2. Jacobian-DeepONet (Extraccion analitica de UQ O(K))
    if args.jacobian_stem:
        print(f"\n[Cargando Modelo] Jacobian-DeepONet: {args.jacobian_stem}")
        if args.variance_activation == "softplus":
            ModelClass = create_jacobian_deeponet_softplus(is_cartesian=True)
        else:
            ModelClass = create_jacobian_deeponet(is_cartesian=True)

        net_jac, scalers_jac = load_pytorch_model(
            model_class=ModelClass,
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
        res_jac = benchmark_model_latency(
            net=net_jac,
            branch=branch,
            trunk=trunk,
            scalers=scalers_jac,
            device=args.device,
            batch_size=args.infer_batch_size,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats
        )
        benchmark_results["Jacobian-DeepONet"] = res_jac

    # 3. Vanilla Pinball DeepONet (Ablacion de bandas uniformes)
    if args.vanilla_stem:
        print(f"\n[Cargando Modelo] Vanilla Pinball: {args.vanilla_stem}")
        VanillaModelClass = create_vanilla_pinball_deeponet(is_cartesian=True)
        net_van, scalers_van = load_pytorch_model(
            model_class=VanillaModelClass,
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
        res_van = benchmark_model_latency(
            net=net_van,
            branch=branch,
            trunk=trunk,
            scalers=scalers_van,
            device=args.device,
            batch_size=args.infer_batch_size,
            n_warmup=args.n_warmup,
            n_repeats=args.n_repeats
        )
        benchmark_results["Vanilla Pinball"] = res_van

    # Reporte consolidado
    print_latency_report(args.equation_name, args.device, dataset_info, benchmark_results)


if __name__ == "__main__":
    main()
