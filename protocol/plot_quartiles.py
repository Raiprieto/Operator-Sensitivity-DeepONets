"""
Descomposicion por cuartiles de error: banda, prediccion y verdad en un corte 1D.

    python protocol/plot_quartiles.py --source trained --out fig.png
    python protocol/plot_quartiles.py --source jcb     --out fig.png

Una fila por benchmark y una columna por cuartil de dificultad. Las muestras se
ordenan por su MSE y se eligen las de los percentiles 25, 50, 75 y 95, de modo
que las columnas recorren de lo facil a lo dificil sin promediar nada.

Dos fuentes de banda, con el mismo centro en cada caso salvo por el modelo:

  trained  el Jacobian-DeepONet entrenado (lambda = 4, semilla 0) y su banda
  jcb      el operador determinista congelado, con la banda post-hoc del
           jacobiano calibrada por conformal. El multiplicador q y la forma se
           leen de jacobian_conformal.json para que la figura y las tablas
           muestren exactamente la misma banda.

El corte 1D se deduce de las coordenadas del trunk, no se asume: se toma la
segunda coordenada mas cercana a su mediana (en Burgers, que es espacio-tiempo,
eso da un instante intermedio) y se dibuja la primera coordenada completa.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402
from jacobian_conformal import shapes  # noqa: E402
from jacobian_sigma import branch_jacobian_sigma  # noqa: E402

# (etiqueta, config entrenada, config determinista, modelo entrenado)
BENCH = [("Burgers 1D", "burgers", "burgers_mse", "jacobian"),
         ("Darcy 2D", "darcy_small", "darcy_small_mse", "jacobian"),
         ("Navier--Stokes 2D", "ns2d_n20k", "ns2d_n20k_mse", "jacobian")]
CUANTILES = (0.25, 0.50, 0.75, 0.95)


def cargar(cfg_nombre, modelo, lam, seed, device):
    cfg = C.load_config(cfg_nombre)
    man = C.load_manifest(cfg)
    C.require_data_matches_manifest(cfg, man, ("test",))
    n = cfg["data"]["splits"]["test"]["n"]
    b_raw, t, y = C.load_split(C.split_path(cfg, "test"), cfg["data"]["append_params"], n)
    d = os.path.join(C.runs_dir(cfg), C.run_name(modelo, lam, seed))
    with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
        m = json.load(f)
    b = C.transform_branch(b_raw, m["config"], m["architecture"]["branch"][0])
    net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
    net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["model_state_dict"])
    net.eval()
    net.unc_ref = m["unc_ref"]
    sc = dict(np.load(os.path.join(d, "scalers.npz")))
    batch = cfg["train"]["val_infer_batch"]
    return cfg, d, net, b, t, y, sc, batch


def corte(t):
    """Indices de una linea 1D: segunda coordenada fija en su mediana."""
    c0, c1 = t[:, 0], t[:, 1]
    valores = np.unique(c1)
    fijo = valores[len(valores) // 2]
    idx = np.where(c1 == fijo)[0]
    return idx[np.argsort(c0[idx])], float(fijo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("trained", "jcb"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C.require_slurm()
    C.require_clean_code()
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(BENCH), len(CUANTILES),
                             figsize=(3.3 * len(CUANTILES), 2.5 * len(BENCH)))
    for fila, (etiqueta, cfg_tr, cfg_det, modelo) in enumerate(BENCH):
        if a.source == "trained":
            cfg, d, net, b, t, y, sc, batch = cargar(cfg_tr, modelo, 4.0, a.seed, device)
            c, lo, up = C.predict(net, b, t, sc, device, batch)
        else:
            cfg, d, net, b, t, y, sc, batch = cargar(cfg_det, "mse", 0.0, a.seed, device)
            c, _, _ = C.predict(net, b, t, sc, device, batch)
            with open(os.path.join(d, "jacobian_conformal.json"), encoding="utf-8") as f:
                jc = json.load(f)
            _, s_norm = branch_jacobian_sigma(net, b, t, sc, device)
            h = shapes(s_norm, sc["Y_std"], jc["beta"])["jac"]
            q = jc["shapes"]["jac"]["q"]
            lo, up = c - q * h, c + q * h

        mse_s = ((y - c) ** 2).mean(1)
        orden = np.argsort(mse_s)
        linea, fijo = corte(t)
        x = t[linea, 0]
        for col, qq in enumerate(CUANTILES):
            i = orden[min(int(qq * len(orden)), len(orden) - 1)]
            ax = axes[fila, col]
            ax.fill_between(x, lo[i][linea], up[i][linea], color="0.75",
                            label="90% interval" if (fila == 0 and col == 0) else None)
            ax.plot(x, y[i][linea], "k--", lw=1.1,
                    label="ground truth" if (fila == 0 and col == 0) else None)
            ax.plot(x, c[i][linea], color="tab:blue", lw=1.1,
                    label="prediction" if (fila == 0 and col == 0) else None)
            if fila == 0:
                ax.set_title("$Q_{%d}$ (%d%%)" % (col + 1, int(qq * 100)), fontsize=10)
            if col == 0:
                ax.set_ylabel(etiqueta, fontsize=9)
            ax.tick_params(labelsize=7)
        print("  %-20s corte en la coordenada 2 = %.4g, %d nodos" % (etiqueta, fijo, len(linea)),
              flush=True)

    fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight")
    print("escrito", a.out)


if __name__ == "__main__":
    main()
