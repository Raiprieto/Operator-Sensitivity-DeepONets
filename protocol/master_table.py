"""
Tabla maestra: los diseños de UQ comparados en los tres benchmarks.

    python protocol/master_table.py [--md ARCHIVO]

Todos los números salen de los JSON que los produjeron, nunca copiados a mano:

    Darcy y NS2D   runs/protocol/*/*/jacobian_conformal.json   (3 semillas)
    Burgers        runs/legacy/burgers_conformal.json          (1 semilla)

Los ocho diseños comparten centro, partición de calibración, score y piso; lo
único que cambia entre filas es CÓMO se construye el semiancho, porque el
escalar conformal q absorbe cualquier factor global. Por eso las filas son
comparables entre sí dentro de cada benchmark.

Burgers usa los modelos previos al protocolo (una corrida por diseño, sin
manifiesto) y todavía no tiene el diseño C; se marca como tal.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

# (clave, descripcion, modelo, forma, entrena cabeza de UQ)
DISENOS = [
    ("A",  "Determinista + conformal global",        "mse",         "const_phys", "no"),
    ("A'", "Determinista + conformal x Y_std",       "mse",         "const_norm", "no"),
    ("B",  "Determinista + JACOBIANO + conformal",   "mse",         "jac",        "no"),
    ("B'", "Determinista + jacobiano log + conf.",   "mse",         "jac_log",    "no"),
    ("V",  "Vanilla pinball (ancho constante)",      "vanilla",     "own",        "si"),
    ("C",  "Quantile condicionado + conformal",      "quantile_ux", "own",        "si"),
    ("D",  "Jacobian-DeepONet + conformal",          "jacobian",    "own",        "si"),
    ("D'", "Jacobian-DeepONet, geometria recalib.",  "jacobian",    "jac_log",    "si"),
]

BENCHMARKS = [
    ("Darcy 2D", {"mse": "darcy_small_mse", "vanilla": "darcy_small_baselines",
                  "quantile_ux": "darcy_small_baselines", "jacobian": "darcy_small"}),
    ("Navier-Stokes 2D", {"mse": "ns2d_n20k_mse", "vanilla": "ns2d_n20k_baselines",
                          "quantile_ux": "ns2d_n20k_baselines", "jacobian": "ns2d_n20k"}),
    ("Burgers 1D", "legacy"),
]
# Como se llaman esos mismos modelos en la corrida previa al protocolo
LEGACY = {"mse": "determinista", "vanilla": "vanilla_lam4", "jacobian": "jacobian_lam4"}

CAMPOS = ("picp", "mpiw", "interval_score", "rho_err", "rho_sens")

# Darcy es el unico cuyo test NO lleva jacobian_reference: su sensibilidad de
# referencia vive en el conjunto aparte ood_sens_ref. Se recupera de los JSON
# que ya la midieron sobre ese conjunto, aprovechando que rho es invariante a
# la escala y que cada forma coincide con una variante ya calculada:
#
#   forma 'own'  -> ood_sens_ref.json        (el ancho del propio modelo)
#   const_norm   -> ood_sens_ref_sigma.json, variante 'ystd'
#   jac_log      -> ood_sens_ref_sigma.json, variante 'band_proxy'
#
# 'jac' (= Y_std * sigma_norm) no coincide con ninguna variante medida y queda
# sin dato; el resumen usa jac_log, que si lo tiene.
SENS_SIGMA = {"const_norm": "ystd", "jac_log": "band_proxy"}


def rho_sens_aparte(cfg, modelo, forma):
    """rho_Sens de Darcy, desde el conjunto de sensibilidad."""
    vals = []
    for d in sorted(glob.glob(os.path.join(C.REPO, "runs", "protocol", cfg, f"{modelo}_lam*"))):
        if forma == "own":
            p, clave = os.path.join(d, "ood_sens_ref.json"), None
        elif forma in SENS_SIGMA:
            p, clave = os.path.join(d, "ood_sens_ref_sigma.json"), SENS_SIGMA[forma]
        else:
            return []
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        if clave is None:
            if "rho_sens_spearman" in j:
                vals.append(j["rho_sens_spearman"])
        elif clave in j.get("rho_sens", {}):
            vals.append(j["rho_sens"][clave])
    return vals


def del_protocolo(cfg, modelo, forma):
    """Valores por semilla de una celda, desde los JSON del protocolo."""
    vals = {k: [] for k in CAMPOS}
    for f in sorted(glob.glob(os.path.join(C.REPO, "runs", "protocol", cfg,
                                           f"{modelo}_lam*", "jacobian_conformal.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d["model"] != modelo or forma not in d["shapes"]:
            continue
        for k in CAMPOS:
            if k in d["shapes"][forma]:
                vals[k].append(d["shapes"][forma][k])
    return vals


def del_legacy(modelo, forma, cache={}):
    if "filas" not in cache:
        p = os.path.join(C.REPO, "runs", "legacy", "burgers_conformal.json")
        if not os.path.exists(p):
            cache["filas"] = []
        else:
            with open(p, encoding="utf-8") as fh:
                cache["filas"] = json.load(fh)["filas"]
    nombre = LEGACY.get(modelo)
    vals = {k: [] for k in CAMPOS}
    for r in cache["filas"]:
        if r["modelo"] == nombre and r["forma"] == forma:
            for k in CAMPOS:
                if k in r:
                    vals[k].append(r[k])
    return vals


def fmt(v, dec=3, cientifico=False):
    if not v:
        return "--"
    a = np.array(v, dtype=float)
    m = a.mean()
    txt = f"{m:.{dec}e}" if cientifico else f"{m:.{dec}f}"
    if len(a) > 1:
        s = a.std(ddof=1)
        txt += f" ± {s:.{dec}e}" if cientifico else f" ± {s:.{dec}f}"
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", help="Ademas de imprimir, escribe la tabla en este archivo")
    a = ap.parse_args()

    out = []
    p = out.append
    p("# Tabla maestra: diseños de UQ en los tres benchmarks\n")
    p("Todas las filas comparten centro, partición de calibración, score conformal y")
    p("piso relativo; solo cambia cómo se construye el semiancho. El escalar `q` absorbe")
    p("cualquier factor de escala global, así que la comparación es sobre la FORMA.\n")
    p("| Clave | Diseño | ¿Entrena cabeza de UQ? |")
    p("|---|---|---|")
    for clave, desc, _, _, entrena in DISENOS:
        p(f"| **{clave}** | {desc} | {'**no**' if entrena == 'no' else 'sí'} |")
    p("")

    resumen = {}
    for etiqueta, fuente in BENCHMARKS:
        legacy = fuente == "legacy"
        p(f"\n## {etiqueta}" + ("  *(modelos previos al protocolo, 1 semilla)*" if legacy else
                                "  *(protocolo, 3 semillas: media ± desv. muestral)*") + "\n")
        p("| Clave | Diseño | PICP (%) | MPIW@90% | vs A' | IS | ρErr | ρSens |")
        p("|---|---|---|---|---|---|---|---|")
        ref = None
        celdas = {}
        for clave, desc, modelo, forma, _ in DISENOS:
            v = del_legacy(modelo, forma) if legacy else del_protocolo(fuente[modelo], modelo, forma)
            if not legacy and not v["rho_sens"]:
                v["rho_sens"] = rho_sens_aparte(fuente[modelo], modelo, forma)
            celdas[clave] = v
            if clave == "A'" and v["mpiw"]:
                ref = float(np.mean(v["mpiw"]))
        for clave, desc, modelo, forma, _ in DISENOS:
            v = celdas[clave]
            if not v["mpiw"]:
                p(f"| **{clave}** | {desc} | -- | -- | -- | -- | -- | -- |")
                continue
            rel = float(np.mean(v["mpiw"])) / ref if ref else float("nan")
            p(f"| **{clave}** | {desc} | {fmt(v['picp'], 2)} | {fmt(v['mpiw'], 3, True)} | "
              f"{rel:.3f}x | {fmt(v['interval_score'], 3, True)} | {fmt(v['rho_err'])} | "
              f"{fmt(v['rho_sens'])} |")
        resumen[etiqueta] = celdas

    # ------------------------------------------------------------------
    p("\n\n## Resumen: las tres preguntas\n")
    p("Todo relativo a **A'**, el mejor baseline que no entrena nada (ancho constante")
    p("en espacio normalizado, que es exactamente lo que aprende el `vanilla`).\n")
    p("| Pregunta | " + " | ".join(e for e, _ in BENCHMARKS) + " |")
    p("|---|" + "---|" * len(BENCHMARKS))

    def linea(titulo, fn):
        p(f"| {titulo} | " + " | ".join(fn(resumen[e]) for e, _ in BENCHMARKS) + " |")

    def med(c, clave, campo):
        v = c[clave][campo]
        return float(np.mean(v)) if v else float("nan")

    def ratio(c, num, den, campo="mpiw"):
        return f"{med(c, num, campo) / med(c, den, campo):.3f}x"

    def flecha(c, desde, hasta, campo):
        if np.isnan(med(c, desde, campo)):
            return "--"
        return f"{med(c, desde, campo):.3f} -> **{med(c, hasta, campo):.3f}**"

    linea("**1.** Ancho: jacobiano congelado vs constante (B' vs A')",
          lambda c: ratio(c, "B'", "A'"))
    linea("&nbsp;&nbsp;&nbsp;rhoErr: A' -> B'", lambda c: flecha(c, "A'", "B'", "rho_err"))
    linea("&nbsp;&nbsp;&nbsp;rhoSens: A' -> B'", lambda c: flecha(c, "A'", "B'", "rho_sens"))
    def versus(c, izq, der, campo, dec=3, cientifico=False):
        x, y = med(c, izq, campo), med(c, der, campo)
        if np.isnan(x):
            return "--"
        f = "e" if cientifico else "f"
        return f"**{x:.{dec}{f}}** vs {y:.{dec}{f}}"

    linea("**2.** rhoSens: congelado (B') vs entrenado (D)",
          lambda c: versus(c, "B'", "D", "rho_sens"))
    linea("**3.** Cabeza entrenada (D) vs su geometria recalibrada (D')",
          lambda c: versus(c, "D", "D'", "mpiw", cientifico=True))

    texto = "\n".join(out)
    print(texto)
    if a.md:
        with open(a.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto + "\n")
        print(f"\n[escrito en {a.md}]")


if __name__ == "__main__":
    main()
