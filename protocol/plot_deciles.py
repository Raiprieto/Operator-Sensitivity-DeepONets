"""
Cobertura y ancho por decil de dificultad, una figura por benchmark.

    python protocol/plot_deciles.py --benchmark darcy --out fig.png

Las muestras de test se ordenan por su propio MSE y se parten en diez grupos;
common.metrics ya escribe decile_picp y decile_mpiw en cada evaluacion, asi que
esto solo lee JSON y no necesita GPU ni datos.

Panel izquierdo: cobertura por decil, con la linea nominal del 90 %. Panel
derecho: ancho medio por decil. La banda sombreada es la desviacion sobre las
tres semillas, no un intervalo de confianza.

Todo se reporta TAL COMO SE ENTRENO, sin rescalado conformal: el punto de la
figura es como cada parametrizacion reparte el ancho por si sola, y calibrar
despues borraria justamente eso.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

# (etiqueta, config, modelo, color, estilo)
BENCH = {
    "burgers": ("Burgers 1D", [
        ("Constant-width pinball", "burgers_baselines", "vanilla", "tab:orange", "--"),
        ("Spatial quantile", "burgers_baselines", "quantile_x", "tab:green", ":"),
        ("Conditional quantile", "burgers_baselines", "quantile_ux", "tab:red", "-."),
        ("Jacobian-DeepONet", "burgers", "jacobian", "tab:blue", "-")]),
    "darcy": ("Darcy 2D", [
        ("Constant-width pinball", "darcy_small_baselines", "vanilla", "tab:orange", "--"),
        ("Spatial quantile", "darcy_small_baselines", "quantile_x", "tab:green", ":"),
        ("Conditional quantile", "darcy_small_baselines", "quantile_ux", "tab:red", "-."),
        ("Jacobian-DeepONet", "darcy_small", "jacobian", "tab:blue", "-")]),
    "ns2d": ("Navier-Stokes 2D", [
        ("Constant-width pinball", "ns2d_n20k_baselines", "vanilla", "tab:orange", "--"),
        ("Spatial quantile", "ns2d_n20k_baselines", "quantile_x", "tab:green", ":"),
        ("Conditional quantile", "ns2d_n20k_baselines", "quantile_ux", "tab:red", "-."),
        ("Jacobian-DeepONet", "ns2d_n20k", "jacobian", "tab:blue", "-")]),
}


def deciles(cfg, modelo):
    """(picp, mpiw) apilados por semilla, cada uno [n_semillas, 10]."""
    p, w = [], []
    for d in sorted(glob.glob(os.path.join(C.REPO, "runs", "protocol", cfg, "%s_lam*" % modelo))):
        f = os.path.join(d, "test_metrics.json")
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            t = json.load(fh)["test"]
        if "decile_picp" in t:
            p.append(t["decile_picp"])
            w.append(t["decile_mpiw"])
    return np.array(p), np.array(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(BENCH))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    etiqueta, filas = BENCH[a.benchmark]
    x = np.arange(1, 11)
    fig, (ax_p, ax_w) = plt.subplots(1, 2, figsize=(10, 3.6))

    for nombre, cfg, modelo, color, estilo in filas:
        p, w = deciles(cfg, modelo)
        if not len(p):
            print("  sin datos:", cfg, modelo)
            continue
        for ax, v in ((ax_p, p), (ax_w, w)):
            m, s = v.mean(0), (v.std(0, ddof=1) if len(v) > 1 else np.zeros(v.shape[1]))
            ax.plot(x, m, estilo, color=color, lw=1.6, marker="o", ms=3.5, label=nombre)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
        print("  %-24s n=%d  PICP %5.1f -> %5.1f  MPIW x%.2f"
              % (nombre, len(p), p.mean(0)[0], p.mean(0)[-1], w.mean(0)[-1] / w.mean(0)[0]), flush=True)

    ax_p.axhline(90, color="0.4", lw=1.0, ls=(0, (4, 3)))
    ax_p.text(10.2, 90, "nominal", va="center", fontsize=8, color="0.4")
    ax_p.set_ylabel("coverage (%)")
    ax_w.set_ylabel("mean interval width")
    ax_w.set_yscale("log")
    for ax in (ax_p, ax_w):
        ax.set_xlabel("error decile ($D_1$ easiest $\\rightarrow$ $D_{10}$ hardest)")
        ax.set_xticks(x)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    fig.suptitle(etiqueta, fontsize=11)
    handles, labels = ax_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("escrito", a.out)


if __name__ == "__main__":
    main()
