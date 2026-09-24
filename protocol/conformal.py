"""
Rescala los intervalos de cada corrida para compararlos a COBERTURA IGUAL.

    python protocol/conformal.py <benchmark> [--seed N] [--lam L] [--smoke]

Problema que resuelve: dos modelos con coberturas distintas no se pueden
comparar por ancho. Uno puede ser mas angosto simplemente porque cubre menos.

Metodo (split conformal multiplicativo). El test se parte en dos mitades
disjuntas por muestra, con una particion fija (semilla 0) que NO depende del
modelo: calibracion y evaluacion. Sobre la mitad de calibracion se mide, en
cada punto, por cuanto habria que multiplicar la semibanda para que el valor
real quedara justo en el borde,

    s = max( (c - y) / (c - lo),  (y - c) / (up - c) )

(s <= 1 significa que el punto ya estaba cubierto, s > 1 que se quedo fuera)
y se toma el cuantil (1 - alpha) de esos scores. Ese unico escalar q reescala
las dos mitades de la banda alrededor del centro,

    lo' = c - q (c - lo),      up' = c + q (up - c),

y las metricas se reportan SOLO sobre la mitad de evaluacion, que no participo
de la calibracion. Asi todos los modelos quedan en ~90 % de cobertura y sus
anchos son comparables.

Advertencia honesta sobre la garantia: el cuantil se calcula punto a punto, de
modo que iguala el PICP tal como lo define el protocolo (fraccion de puntos
cubiertos). Los puntos de una misma muestra no son intercambiables entre si,
asi que esto es una calibracion empirica del ancho, no la garantia formal de
cobertura marginal del conformal clasico (que exigiria un score por muestra).

El modelo determinista se omite: no tiene banda que reescalar.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402

SPLIT_SEED = 0          # particion calibracion/evaluacion, fija e igual para todos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lam", type=float, help="Solo corridas con este lambda")
    ap.add_argument("--seed", type=int, help="Solo corridas con esta semilla")
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    cfg = C.load_config(a.benchmark, a.smoke)
    man = C.load_manifest(cfg)
    C.require_data_matches_manifest(cfg, man, ("test",))
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    n_te = cfg["data"]["splits"]["test"]["n"]
    bte_raw, t, yte = C.load_split(C.split_path(cfg, "test"), cfg["data"]["append_params"], n_te)
    batch = cfg["train"]["val_infer_batch"]

    # Particion fija por muestra: la misma para todas las corridas y modelos
    idx = np.random.default_rng(SPLIT_SEED).permutation(n_te)
    cal, ev = np.sort(idx[: n_te // 2]), np.sort(idx[n_te // 2:])

    rows = []
    for kind, lam, seed in C.plan(cfg):
        if (a.lam is not None and lam != a.lam) or (a.seed is not None and seed != a.seed):
            continue
        d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
        mp = os.path.join(d, "metadata.json")
        if not os.path.exists(mp):
            print(f"  {os.path.basename(d):28s} omitida: no existe")
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m["status"] != "completed":
            print(f"  {os.path.basename(d):28s} omitida: status={m['status']}")
            continue
        if m["model"] == "mse":
            print(f"  {os.path.basename(d):28s} omitida: determinista, no tiene banda")
            continue
        if m["data_md5"] != {s: man["files"][s]["md5"] for s in C.SPLITS}:
            raise C.ProtocolError(f"{d} se entreno con otros datos que los del manifiesto.")
        out = os.path.join(d, "conformal.json")
        C.require_absent(out)

        bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
        net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["model_state_dict"])
        net.eval()
        net.unc_ref = m["unc_ref"]
        sc = dict(np.load(os.path.join(d, "scalers.npz")))
        c, lo, up = C.predict(net, bte, t, sc, device, batch)

        half_lo = np.maximum(c - lo, 1e-12)
        half_up = np.maximum(up - c, 1e-12)
        # Factor por el que hay que multiplicar la semibanda para tocar el punto:
        # se mide desde el CENTRO, no desde el borde (s = 1 es la banda original).
        s = np.maximum((c - yte) / half_lo, (yte - c) / half_up)
        q = float(np.quantile(s[cal], 1 - C.ALPHA))
        lo_q, up_q = c - q * half_lo, c + q * half_up

        res = {"q": q, "n_cal": len(cal), "n_eval": len(ev), "split_seed": SPLIT_SEED,
               "as_trained_eval": C.metrics(yte[ev], c[ev], lo[ev], up[ev]),
               "conformal_eval": C.metrics(yte[ev], c[ev], lo_q[ev], up_q[ev]),
               "conformal_cal": C.metrics(yte[cal], c[cal], lo_q[cal], up_q[cal]),
               "commit_eval": commit, "test_md5": man["files"]["test"]["md5"]}
        if res["conformal_eval"]["nonfinite"]:
            raise C.ProtocolError(f"{d}: metricas conformales no finitas")
        C.write_json(out, res)
        rows.append((kind, lam, seed, res))
        print(f"  {os.path.basename(d):28s} q={q:6.3f}  PICP {res['as_trained_eval']['picp']:6.2f}% "
              f"-> {res['conformal_eval']['picp']:6.2f}%   MPIW {res['as_trained_eval']['mpiw']:.4e} "
              f"-> {res['conformal_eval']['mpiw']:.4e}", flush=True)

    print("\n| Modelo | λ | Semilla | q | PICP as-trained | PICP conformal | MPIW@90% | IS |")
    print("|---|---|---|---|---|---|---|---|")
    for kind, lam, seed, r in rows:
        print(f"| {kind} | {lam:g} | {seed} | {r['q']:.3f} | {r['as_trained_eval']['picp']:.2f} | "
              f"{r['conformal_eval']['picp']:.2f} | {r['conformal_eval']['mpiw']:.4e} | "
              f"{r['conformal_eval']['interval_score']:.4e} |")


if __name__ == "__main__":
    main()
