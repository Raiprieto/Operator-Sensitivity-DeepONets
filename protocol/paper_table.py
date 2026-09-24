"""
Emite las filas LaTeX de la Tabla 1 del paper desde los resultados del protocolo.

    python protocol/paper_table.py                 (en el cluster)

Cada numero sale del JSON que lo produjo, nunca copiado a mano:

    MSE, PICP    test_metrics.json      (test completo, tal como se entreno)
    MPIW@90%     conformal.json         (mitad de evaluacion, tras el rescalado)
    rho_Err      test_correlations.json (test completo)
    rho_Sens     test_correlations.json, o el conjunto de sensibilidad en Darcy

Se reporta media +- desviacion muestral (ddof=1) sobre las semillas, igual que
protocol/aggregate.py. El determinista solo tiene MSE: sin banda no hay PICP,
MPIW ni correlaciones que reportar.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

# (etiqueta, (escala, decimales) de MSE, idem de MPIW, de donde sale rho_Sens, filas)
# Las escalas son las que ya usa la tabla del paper; los decimales, los que hacen
# falta para que las filas no se vean iguales por redondeo.
# Cada fila: (nombre en el paper, config, modelo)
BLOQUES = [
    ("Darcy 2D", (1e-6, 2), (1e-3, 2), "ood_sens_ref.json", [
        ("Deterministic", "darcy_small_mse", "mse"),
        ("Conditional quantile", "darcy_small_baselines", "quantile_ux"),
        ("Jacobian-DeepONet", "darcy_small", "jacobian"),
    ]),
    ("Navier--Stokes 2D", (1e-3, 3), (1e-1, 3), "test_correlations.json", [
        ("Deterministic", "ns2d_n20k_mse", "mse"),
        ("Conditional quantile", "ns2d_n20k_baselines", "quantile_ux"),
        ("Jacobian-DeepONet", "ns2d_n20k", "jacobian"),
    ]),
]


def leer(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recolectar(bench, modelo, sens_file):
    """Valores por semilla de una fila de la tabla."""
    cfg = C.load_config(bench)
    vals = {k: [] for k in ("mse", "picp", "mpiw90", "rho_err", "rho_sens")}
    for kind, lam, seed in C.plan(cfg):
        if kind != modelo:
            continue
        d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
        t = leer(os.path.join(d, "test_metrics.json"))
        if t is None:
            raise SystemExit(f"Falta test_metrics.json en {d}")
        vals["mse"].append(t["test"]["mse"])
        if modelo != "mse":          # sin banda no hay cobertura que reportar
            vals["picp"].append(t["test"]["picp"])
        cf = leer(os.path.join(d, "conformal.json"))
        if cf:
            vals["mpiw90"].append(cf["conformal_eval"]["mpiw"])
        co = leer(os.path.join(d, "test_correlations.json"))
        if co and "rho_err_spearman" in co:
            vals["rho_err"].append(co["rho_err_spearman"])
        sens = co if sens_file == "test_correlations.json" else leer(os.path.join(d, sens_file))
        if sens and "rho_sens_spearman" in sens:
            vals["rho_sens"].append(sens["rho_sens_spearman"])
    return vals


def ms(v, escala=1.0, dec=2):
    """media ± desviacion muestral, o '--' si no aplica."""
    if not v:
        return "--"
    a = np.array(v, dtype=float) / escala
    s = a.std(ddof=1) if len(a) > 1 else 0.0
    return f"${a.mean():.{dec}f} \\pm {s:.{dec}f}$"


def main():
    for etiqueta, (esc_mse, dec_mse), (esc_mpiw, dec_w), sens_file, filas in BLOQUES:
        e_mse = f"{int(round(np.log10(esc_mse)))}"
        e_w = f"{int(round(np.log10(esc_mpiw)))}"
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{{etiqueta} "
              f"(MSE $\\times 10^{{{e_mse}}}$, MPIW $\\times 10^{{{e_w}}}$)}}}} \\\\")
        for nombre, bench, modelo in filas:
            v = recolectar(bench, modelo, sens_file)
            n = len(v["mse"])
            print(f"{nombre} & {ms(v['mse'], esc_mse, dec_mse)} & {ms(v['picp'], 1.0, 1)} & "
                  f"{ms(v['mpiw90'], esc_mpiw, dec_w)} & {ms(v['rho_err'], 1.0, 3)} & "
                  f"{ms(v['rho_sens'], 1.0, 3)} \\\\"
                  + (f"   % {n} semillas" if n != 3 else ""))
        print("\\midrule")


if __name__ == "__main__":
    main()
