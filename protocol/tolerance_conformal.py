"""
PRUEBA DE CONCEPTO: region de tolerancia conformal, con garantia de verdad.

    python protocol/tolerance_conformal.py <benchmark> --seed N
    python protocol/tolerance_conformal.py --legacy            (Burgers previo al protocolo)

El problema que arregla. protocol/jacobian_conformal.py toma el cuantil sobre
los scores de TODOS los nodos agrupados. Eso exigiria que las n x N_nodos
unidades fueran intercambiables, y no lo son: los nodos de un mismo campo estan
correlacionados y ni siquiera comparten distribucion marginal (la frontera de
Dirichlet contra el centro). El tamano muestral efectivo es n, no n x N_nodos,
y el cuantil agrupado no hereda ninguna garantia.

Lo que si es intercambiable son las MUESTRAS: mismo generador, rangos de semilla
disjuntos, verificado por KS en el manifiesto. Basta entonces con colapsar cada
campo a un escalar antes de tomar el cuantil. Usamos el cuantil sobre nodos, no
el maximo, porque el maximo sobre 12800 nodos da bandas inutilmente anchas:

    s^(k) = Quantile_{1-gamma, i} ( |y_i - c_i| / h(u, x_i) )

y sobre esos n escalares tomamos el ESTADISTICO DE ORDEN ceil((n+1)(1-alpha)),
no el cuantil empirico: la correccion (n+1) es justamente lo que da la garantia
exacta en muestra finita. El resultado es una region de tolerancia,

    P( al menos una fraccion 1-gamma de los nodos de una funcion nueva
        cae dentro de la banda ) >= 1 - alpha

que con gamma = alpha = 0.1 se lee "el 90 % del campo cubierto, en el 90 % de
los casos". Esta garantia si se sostiene, a diferencia del PICP agrupado.

Se reportan las dos calibraciones lado a lado sobre la MISMA mitad de
evaluacion, para poder medir cuanto ancho cuesta la garantia:

    puntual     q del cuantil agrupado por nodo (lo que hace el paper hoy)
    tolerancia  q del estadistico de orden sobre scores por muestra

La columna que valida el metodo es `exito_muestral`: la fraccion de funciones de
evaluacion con al menos 1-gamma de sus nodos cubiertos. Tiene que dar >= 1-alpha.

Nota: la intercambiabilidad vale entre calibracion y test porque salen del mismo
generador. Se rompe por construccion en los conjuntos OOD, donde ninguna de las
dos calibraciones garantiza nada.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402
from conformal import SPLIT_SEED  # noqa: E402
from jacobian_conformal import shapes  # noqa: E402
from jacobian_sigma import branch_jacobian_sigma  # noqa: E402

GAMMA = 0.10        # fraccion de nodos que se permite fuera de la banda


def q_tolerancia(err, h, cal, gamma, alpha):
    """Estadistico de orden ceil((n+1)(1-alpha)) de los scores por muestra.

    Devuelve (q, indice_usado, n_cal). q = inf significa que con este numero de
    muestras de calibracion la garantia no es alcanzable.
    """
    s = np.quantile((err / h)[cal], 1 - gamma, axis=1)      # un escalar por muestra
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf"), k, n
    return float(np.sort(s)[k - 1]), k, n


def evaluar(yte, c, h, q, ev, gamma):
    """Metricas sobre la mitad de evaluacion, incluida la garantia propia."""
    lo, up = c - q * h, c + q * h
    m = C.metrics(yte[ev], c[ev], lo[ev], up[ev])
    cubierto = ((yte[ev] >= lo[ev]) & (yte[ev] <= up[ev])).mean(axis=1)
    m["exito_muestral"] = float((cubierto >= 1 - gamma).mean() * 100)
    m["q"] = q
    return m


def procesar(nombre, net, sc, bte, t, yte, cal, ev, batch, beta, gamma, filas):
    c, lo, up = C.predict(net, bte, t, sc, "cuda", batch)
    _, s_norm = branch_jacobian_sigma(net, bte, t, sc, "cuda")
    err = np.abs(yte - c)
    half = (up - lo) / 2
    own = half if np.max(half) > 1e-8 * max(np.abs(c).max(), 1e-30) else None

    for forma, h in shapes(s_norm, sc["Y_std"], beta, own).items():
        qt, k, n = q_tolerancia(err, h, cal, gamma, C.ALPHA)
        if not np.isfinite(qt):
            print(f"  {nombre}/{forma}: {n} muestras no alcanzan para la garantia (hace falta k={k})")
            continue
        qp = float(np.quantile((err / h)[cal], 1 - C.ALPHA))     # calibracion puntual, referencia
        tol = evaluar(yte, c, h, qt, ev, gamma)
        pun = evaluar(yte, c, h, qp, ev, gamma)
        filas.append({"modelo": nombre, "forma": forma, "k": k, "n_cal": n,
                      "tolerancia": tol, "puntual": pun})
        print(f"  {nombre:14s} {forma:11s} q {qp:.4g} -> {qt:.4g}   "
              f"exito muestral {pun['exito_muestral']:5.1f}% -> {tol['exito_muestral']:5.1f}%   "
              f"MPIW x{tol['mpiw'] / pun['mpiw']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", nargs="?")
    ap.add_argument("--legacy", action="store_true", help="Modelos de Burgers previos al protocolo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")

    filas = []
    if a.legacy:
        import legacy_conformal as L
        bte, t, yte = C.load_split(os.path.join(C.REPO, L.TEST), True)
        n_te = len(bte)
        idx = np.random.default_rng(SPLIT_SEED).permutation(n_te)
        cal, ev = np.sort(idx[: n_te // 2]), np.sort(idx[n_te // 2:])
        print(f"legacy Burgers: {n_te} muestras x {yte.shape[1]} nodos | "
              f"cal {len(cal)} / eval {len(ev)} | gamma {a.gamma}", flush=True)
        btr = None
        for nombre, kind, stem in L.MODELOS:
            net, sc = L.cargar(os.path.join(C.REPO, stem), kind, "cuda")
            if kind == "jacobian":
                if btr is None:
                    btr = C.load_split(os.path.join(C.REPO, L.TRAIN), True, L.N_UNC_REF)[0]
                net.unc_ref = C.compute_unc_ref(net, btr, t, sc, "cuda", 50)
            procesar(nombre, net, sc, bte, t, yte, cal, ev, 50, a.beta, a.gamma, filas)
        etiqueta, destino = "burgers_legacy", os.path.join(C.REPO, "runs", "legacy")
    else:
        cfg = C.load_config(a.benchmark)
        man = C.load_manifest(cfg)
        C.require_data_matches_manifest(cfg, man, ("test",))
        n_te = cfg["data"]["splits"]["test"]["n"]
        bte_raw, t, yte = C.load_split(C.split_path(cfg, "test"), cfg["data"]["append_params"], n_te)
        batch = cfg["train"]["val_infer_batch"]
        idx = np.random.default_rng(SPLIT_SEED).permutation(n_te)
        cal, ev = np.sort(idx[: n_te // 2]), np.sort(idx[n_te // 2:])
        print(f"{a.benchmark} semilla {a.seed}: {n_te} muestras x {yte.shape[1]} nodos | "
              f"cal {len(cal)} / eval {len(ev)} | gamma {a.gamma}", flush=True)
        for kind, lam, seed in C.plan(cfg):
            if seed != a.seed:
                continue
            d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
            mp = os.path.join(d, "metadata.json")
            if not os.path.exists(mp):
                continue
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            if m["status"] != "completed":
                continue
            bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
            net = C.build_net(m["architecture"], m["model"], m["sigma"]).to("cuda")
            net.load_state_dict(torch.load(os.path.join(d, "best.pt"),
                                           map_location="cuda")["model_state_dict"])
            net.eval()
            net.unc_ref = m["unc_ref"]
            sc = dict(np.load(os.path.join(d, "scalers.npz")))
            procesar(kind, net, sc, bte, t, yte, cal, ev, batch, a.beta, a.gamma, filas)
        etiqueta = f"{a.benchmark}_seed{a.seed}"
        destino = os.path.join(C.REPO, "runs", "tolerance")

    os.makedirs(destino, exist_ok=True)
    C.write_json(os.path.join(destino, f"tolerance_{etiqueta}.json"),
                 {"commit_eval": commit, "gamma": a.gamma, "alpha": C.ALPHA, "beta": a.beta,
                  "split_seed": SPLIT_SEED, "semillas": 1, "filas": filas})

    print("\n| Modelo | Forma | q puntual | q tolerancia | exito muestral punt. | "
          "exito muestral tol. | PICP tol. | MPIW punt. | MPIW tol. | factor |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f in filas:
        p, t_ = f["puntual"], f["tolerancia"]
        print(f"| {f['modelo']} | {f['forma']} | {p['q']:.4g} | {t_['q']:.4g} | "
              f"{p['exito_muestral']:.1f}% | **{t_['exito_muestral']:.1f}%** | {t_['picp']:.2f} | "
              f"{p['mpiw']:.3e} | {t_['mpiw']:.3e} | {t_['mpiw'] / p['mpiw']:.2f}x |")


if __name__ == "__main__":
    main()
