"""
PRUEBA DE CONCEPTO: comparacion de disenos de UQ sobre los modelos de Burgers
ANTERIORES al protocolo.

    python protocol/legacy_conformal.py [--only NOMBRE]

Por que existe. Burgers no tiene todavia corridas del protocolo, pero si tiene
modelos ya entrenados del paper, y tres de ellos comparten exactamente el mismo
backbone (129 -> 256x10, trunk 2 -> 256x10, swish, K = 256):

    determinista  modelos/burgers/burgers_simple_paper_256      (sin bandas)
    vanilla       .../burgers_burgers_vanilla_pinball_lambda4_256
    jacobian      .../burgers_burgers_lambda4_256

Eso alcanza para la pregunta de fondo: si el jacobiano de un modelo ya
entrenado, extraido del forward y calibrado con conformal, da un intervalo
competitivo sin reentrenar nada.

QUE NO ES. Estos artefactos son previos al protocolo: no tienen manifiesto, ni
metadata firmada, ni tres semillas. Son UNA corrida por diseno, sobre
data/burgers_test.h5 (verificado sin fuga el 21-09: 0 de 500 muestras en el
train, distancia minima al vecino 2.089). Los numeros sirven como prueba de
concepto, no como resultado del protocolo; el equivalente auditado sale de
burgers_mse / burgers_baselines / burgers cuando terminen de entrenar.

Todo lo demas es identico a protocol/jacobian_conformal.py: las mismas formas
de semiancho, el mismo score, la misma particion (SPLIT_SEED) y el mismo piso
relativo beta, de modo que la tabla se lee junto a las de Darcy y NS2D.
"""

import argparse
import os
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402
from conformal import SPLIT_SEED  # noqa: E402
from jacobian_conformal import shapes  # noqa: E402
from jacobian_sigma import branch_jacobian_sigma  # noqa: E402

TEST = "data/burgers_test.h5"
TRAIN = "data/burgers_train_large.h5"
N_UNC_REF = 1024          # mismo subconjunto que unc_ref_subset en las configs
ARQ = {"branch": [129] + [256] * 10, "trunk": [2] + [256] * 10, "activation": "swish"}
SIGMA = 0.025

MODELOS = [
    ("determinista", "mse", "modelos/burgers/burgers_simple_paper_256"),
    ("vanilla_lam4", "vanilla", "modelos/burgers/burgers_burgers_vanilla_pinball_lambda4_256"),
    ("jacobian_lam4", "jacobian", "modelos/burgers/burgers_burgers_lambda4_256"),
]


def cargar(stem, kind, device):
    net = C.build_net(ARQ, kind, SIGMA).to(device)
    ck = torch.load(f"{stem}-200000.pt", map_location=device, weights_only=False)
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    return net, dict(np.load(f"{stem}_scalers.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Evaluar solo este modelo de la lista")
    ap.add_argument("--beta", type=float, default=0.05)
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    bte, t, yte = C.load_split(os.path.join(C.REPO, TEST), True)
    with h5py.File(os.path.join(C.REPO, TEST), "r") as f:
        jac_ref = np.abs(f["jacobian_reference"][:len(bte)])
    n_te = len(bte)
    idx = np.random.default_rng(SPLIT_SEED).permutation(n_te)
    cal, ev = np.sort(idx[: n_te // 2]), np.sort(idx[n_te // 2:])
    print(f"test {n_te} muestras x {yte.shape[1]} nodos | cal {len(cal)} / eval {len(ev)} "
          f"| commit {commit[:8]}", flush=True)

    btr = None
    filas = []
    for nombre, kind, stem in MODELOS:
        if a.only and a.only != nombre:
            continue
        net, sc = cargar(os.path.join(C.REPO, stem), kind, device)
        if kind == "jacobian":
            # El forward jacobiano normaliza la incertidumbre por una referencia.
            # Sin fijarla usa la media del batch y el intervalo deja de ser
            # invariante al tamano de lote; se fija sobre el train, como el protocolo.
            if btr is None:
                btr = C.load_split(os.path.join(C.REPO, TRAIN), True, N_UNC_REF)[0]
            net.unc_ref = C.compute_unc_ref(net, btr, t, sc, device, 50)
            print(f"  unc_ref({nombre}) = {net.unc_ref:.6g} sobre {N_UNC_REF} muestras de train")

        c, lo, up = C.predict(net, bte, t, sc, device, 50)
        c2, lo2, up2 = C.predict(net, bte, t, sc, device, 7)
        drift = (max(np.abs(c - c2).max(), np.abs(lo - lo2).max(), np.abs(up - up2).max())
                 / C.pred_scale(c, lo, up))
        if drift > 1e-4:
            raise C.ProtocolError(f"{nombre}: la prediccion cambia con el tamano de batch ({drift:.2e})")

        _, s_norm = branch_jacobian_sigma(net, bte, t, sc, device)
        err = np.abs(yte - c)
        half = (up - lo) / 2
        own = half if np.max(half) > 1e-8 * max(np.abs(c).max(), 1e-30) else None

        for forma, h in shapes(s_norm, sc["Y_std"], a.beta, own).items():
            q = float(np.quantile((err / h)[cal], 1 - C.ALPHA))
            r = C.metrics(yte[ev], c[ev], c[ev] - q * h[ev], c[ev] + q * h[ev])
            if r["nonfinite"]:
                raise C.ProtocolError(f"{nombre}/{forma}: metricas no finitas")
            r.update({"modelo": nombre, "forma": forma, "q": q,
                      "mse": float((err ** 2).mean())})
            if np.ptp(h[ev]) > 0:
                r["rho_err"] = float(spearmanr(h[ev].ravel(), err[ev].ravel())[0])
                r["rho_sens"] = float(spearmanr(h[ev].ravel(), jac_ref[ev].ravel())[0])
            filas.append(r)
        print(f"  {nombre:14s} listo (MSE {float((err ** 2).mean()):.4e})", flush=True)

    out = os.path.join(C.REPO, "runs", "legacy", "burgers_conformal.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    C.write_json(out, {"commit_eval": commit, "dataset": TEST, "n_test": n_te,
                       "split_seed": SPLIT_SEED, "beta": a.beta, "semillas": 1,
                       "nota": "Artefactos previos al protocolo: una corrida por diseno.",
                       "filas": filas})
    print(f"\nEscrito {os.path.relpath(out, C.REPO)}")

    base = {f["modelo"]: f["mpiw"] for f in filas if f["forma"] == "const_phys"}
    print("\n| Modelo | Forma | q | PICP (%) | MPIW@90% | vs const | IS | rhoErr | rhoSens |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in filas:
        g = lambda k: f"{f[k]:.3f}" if k in f else "-"          # noqa: E731
        print(f"| {f['modelo']} | {f['forma']} | {f['q']:.4g} | {f['picp']:.2f} | "
              f"{f['mpiw']:.4e} | {f['mpiw'] / base[f['modelo']]:.3f} | "
              f"{f['interval_score']:.4e} | {g('rho_err')} | {g('rho_sens')} |")


if __name__ == "__main__":
    main()
