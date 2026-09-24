"""
Cobertura y ancho por decil de dificultad, para comparar disenos de UQ.

    python protocol/deciles_table.py [--benchmark darcy|ns2d|burgers]

Las muestras de la mitad de evaluacion se ordenan por su MSE y se parten en diez
grupos: el decil 1 es donde el modelo acierta y el 10 donde mas se equivoca
(common.metrics ya los calcula en cada evaluacion). Es la prueba que un promedio
global esconde.

Que mirar. Todos los disenos estan calibrados al 90 % GLOBAL sobre la misma
mitad de calibracion, asi que la pregunta no es cuanto cubren en promedio sino
COMO reparten esa cobertura:

  - Un ancho no condicionado gasta lo mismo en todas partes: sobrecubre el
    decil facil y se desploma en el dificil.
  - Un ancho bien condicionado se ensancha donde el error crece y mantiene la
    cobertura plana. La razon MPIW(d10)/MPIW(d1) mide cuanto se adapta.

No necesita GPU ni datos: lee los JSON ya escritos.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

# benchmark -> (etiqueta, [(clave, descripcion, config, modelo, forma)])
GRUPOS = {
    "darcy": ("Darcy 2D", [
        ("A'", "Determinista + conformal x Y_std", "darcy_small_mse", "mse", "const_norm"),
        ("B", "Determinista + JACOBIANO", "darcy_small_mse", "mse", "jac"),
        ("B'", "Determinista + jacobiano log", "darcy_small_mse", "mse", "jac_log"),
        ("C", "Quantile condicionado", "darcy_small_baselines", "quantile_ux", "own"),
        ("D", "Jacobian-DeepONet", "darcy_small", "jacobian", "own"),
    ]),
    "ns2d": ("Navier-Stokes 2D", [
        ("A'", "Determinista + conformal x Y_std", "ns2d_n20k_mse", "mse", "const_norm"),
        ("B", "Determinista + JACOBIANO", "ns2d_n20k_mse", "mse", "jac"),
        ("B'", "Determinista + jacobiano log", "ns2d_n20k_mse", "mse", "jac_log"),
        ("C", "Quantile condicionado", "ns2d_n20k_baselines", "quantile_ux", "own"),
        ("D", "Jacobian-DeepONet", "ns2d_n20k", "jacobian", "own"),
    ]),
}


def recolectar(cfg, modelo, forma):
    """(picp_por_decil, mpiw_por_decil) apilados por semilla."""
    picp, mpiw = [], []
    for f in sorted(glob.glob(os.path.join(C.REPO, "runs", "protocol", cfg,
                                           f"{modelo}_lam*", "jacobian_conformal.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        s = d["shapes"].get(forma)
        if d["model"] != modelo or s is None or "decile_picp" not in s:
            continue
        picp.append(s["decile_picp"])
        mpiw.append(s["decile_mpiw"])
    return np.array(picp), np.array(mpiw)


def legacy(forma, modelo):
    p = os.path.join(C.REPO, "runs", "legacy", "burgers_conformal.json")
    if not os.path.exists(p):
        return np.array([]), np.array([])
    with open(p, encoding="utf-8") as fh:
        filas = json.load(fh)["filas"]
    for r in filas:
        if r["modelo"] == modelo and r["forma"] == forma and "decile_picp" in r:
            return np.array([r["decile_picp"]]), np.array([r["decile_mpiw"]])
    return np.array([]), np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="darcy", choices=list(GRUPOS) + ["burgers"])
    a = ap.parse_args()

    if a.benchmark == "burgers":
        etiqueta = "Burgers 1D (modelos previos al protocolo, 1 semilla)"
        filas = [("A'", "Determinista + conformal x Y_std", "determinista", "const_norm"),
                 ("B", "Determinista + JACOBIANO", "determinista", "jac"),
                 ("B'", "Determinista + jacobiano log", "determinista", "jac_log"),
                 ("D", "Jacobian-DeepONet", "jacobian_lam4", "own")]
        datos = [(k, d, *legacy(f, m)) for k, d, m, f in filas]
    else:
        etiqueta, filas = GRUPOS[a.benchmark]
        datos = [(k, d, *recolectar(cfg, m, f)) for k, d, cfg, m, f in filas]

    print(f"# Cobertura por decil de dificultad — {etiqueta}\n")
    print("Decil 1 = muestras faciles, decil 10 = las que el modelo falla mas.")
    print("Todos calibrados al 90 % global sobre la misma mitad de calibracion.\n")

    print("## PICP (%) por decil\n")
    print("| Diseno | " + " | ".join(f"d{i + 1}" for i in range(10)) + " | d10-d1 |")
    print("|---" * 12 + "|")
    for clave, desc, picp, _ in datos:
        if not len(picp):
            continue
        m = picp.mean(0)
        print(f"| **{clave}** " + "".join(f"| {v:.1f} " for v in m)
              + f"| {m[-1] - m[0]:+.1f} |")

    print("\n## MPIW por decil, normalizado a su propio decil 1\n")
    print("Cuanto se ensancha la banda al aumentar la dificultad. 1.00 en todos")
    print("los deciles = ancho que no se adapta.\n")
    print("| Diseno | " + " | ".join(f"d{i + 1}" for i in range(10)) + " | d10/d1 |")
    print("|---" * 12 + "|")
    for clave, desc, _, mpiw in datos:
        if not len(mpiw):
            continue
        m = mpiw.mean(0)
        r = m / m[0]
        print(f"| **{clave}** " + "".join(f"| {v:.2f} " for v in r) + f"| **{r[-1]:.2f}x** |")

    print("\n## Resumen\n")
    print("| Diseno | Descripcion | PICP d1 | PICP d10 | Caida | MPIW d10/d1 |")
    print("|---|---|---|---|---|---|")
    for clave, desc, picp, mpiw in datos:
        if not len(picp):
            continue
        p, w = picp.mean(0), mpiw.mean(0)
        sd = f" ± {picp.std(0, ddof=1)[-1]:.1f}" if len(picp) > 1 else ""
        print(f"| **{clave}** | {desc} | {p[0]:.1f} | {p[-1]:.1f}{sd} | "
              f"{p[-1] - p[0]:+.1f} | {w[-1] / w[0]:.2f}x |")


if __name__ == "__main__":
    main()
