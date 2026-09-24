"""
Bandas conformales sobre un modelo SIN bandas, moduladas por su propio jacobiano.

    python protocol/jacobian_conformal.py <benchmark> [--seed N] [--lam L] [--beta B]

La idea. Un operador determinista preentrenado no tiene banda, pero si tiene una
geometria: la norma de la fila del jacobiano entrada->salida, que
protocol/jacobian_sigma.py mostro que ya ordena el error sin haber entrenado
nada. Aqui esa geometria se usa como estimador de dificultad de un conformal
normalizado (Papadopoulos 2008; Lei & Wasserman 2014), con score

    s = |y - c| / h(x)_i          h = semiancho propuesto, sin escala

y un unico escalar q = cuantil_(1-alpha) de s sobre la mitad de calibracion. La
banda es c +- q h. Como q absorbe cualquier factor global, lo unico que decide
el resultado es la FORMA de h, que es justo lo que se quiere comparar.

Formas evaluadas, todas sobre el mismo centro y con el mismo procedimiento:

    const_phys  h = 1                      banda constante en unidades fisicas
                                           (el baseline "un solo multiplicador
                                           conformal" del abstract)
    const_norm  h = Y_std                  constante en espacio normalizado;
                                           es la forma que tiene el vanilla
                                           entrenado despues de desnormalizar
    jac         h = Y_std * sigma_norm     geometria del jacobiano
    jac_log     h = Y_std * log1p(sigma_norm / ref)
                                           la forma exacta de la cabeza
                                           entrenada, para separar el efecto de
                                           la compresion logaritmica

sigma_norm es la variante NORMALIZADA de jacobian_sigma: equivale a perturbar
la entrada proporcionalmente a su propia dispersion por coordenada. Pesarla por
1/X_std (unidades fisicas) le da peso a las coordenadas con poca varianza en los
datos y empeora la correlacion con el error; esto es un supuesto explicito, no
un descuido.

Piso. Donde sigma -> 0 (en Darcy la frontera Dirichlet, donde la solucion es
identicamente cero) el cociente |y-c|/h explota, q se infla y la banda queda
sobreancha en TODO el dominio. Por eso h <- h + beta * mediana(h), con beta
declarado. Es el amortiguador estandar del conformal normalizado y no depende
de la escala de h, porque el piso es relativo.

Particion y garantia: las mismas que protocol/conformal.py, de donde se importa
SPLIT_SEED para que las mitades de calibracion y evaluacion sean identicas y los
numeros comparables corrida a corrida. El cuantil es punto a punto, asi que esto
calibra el PICP tal como lo define el protocolo, no la garantia marginal formal.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402
from conformal import SPLIT_SEED  # noqa: E402  (misma particion, a proposito)
from jacobian_sigma import branch_jacobian_sigma  # noqa: E402


def _cronometrar(fn, repeticiones=3):
    """Segundos de la corrida mas rapida, con la GPU sincronizada."""
    mejor = float("inf")
    for _ in range(repeticiones):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def shapes(sigma_norm, ystd, beta, own=None):
    """Semianchos candidatos, ya con el piso relativo aplicado.

    `own` es la semibanda que el propio modelo aprendio, si tiene alguna. Se
    incluye como una forma mas para que las cuatro alternativas pasen por el
    MISMO score, el mismo q y la misma mitad de evaluacion; asi la comparacion
    entre disenos no mezcla procedimientos.
    """
    ys = np.broadcast_to(ystd, sigma_norm.shape)
    ref = float(sigma_norm.mean()) + 1e-30
    out = {"const_phys": np.ones_like(sigma_norm),
           "const_norm": ys.astype(sigma_norm.dtype),
           "jac": ys * sigma_norm,
           "jac_log": ys * np.log1p(sigma_norm / ref)}
    if own is not None:
        out["own"] = own
    return {k: v + beta * float(np.median(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lam", type=float, help="Solo corridas con este lambda")
    ap.add_argument("--seed", type=int, help="Solo corridas con esta semilla")
    ap.add_argument("--beta", type=float, default=0.05,
                    help="Piso relativo del semiancho (fraccion de su mediana)")
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
    with h5py.File(C.split_path(cfg, "test"), "r") as f:
        jac_ref = f["jacobian_reference"][:n_te] if "jacobian_reference" in f else None
    if jac_ref is not None:
        jac_ref = np.abs(jac_ref.sum(axis=2) if jac_ref.ndim == 3 else jac_ref)
    batch = cfg["train"]["val_infer_batch"]

    idx = np.random.default_rng(SPLIT_SEED).permutation(n_te)
    cal, ev = np.sort(idx[: n_te // 2]), np.sort(idx[n_te // 2:])

    rows = []
    for kind, lam, seed in C.plan(cfg):
        if (a.lam is not None and lam != a.lam) or (a.seed is not None and seed != a.seed):
            continue
        d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
        mp = os.path.join(d, "metadata.json")
        if not os.path.exists(mp):
            print(f"  {os.path.basename(d):26s} omitida: no existe")
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m["status"] != "completed":
            print(f"  {os.path.basename(d):26s} omitida: status={m['status']}")
            continue
        if m["data_md5"] != {s: man["files"][s]["md5"] for s in C.SPLITS}:
            raise C.ProtocolError(f"{d} se entreno con otros datos que los del manifiesto.")
        out = os.path.join(d, "jacobian_conformal.json")
        C.require_absent(out)

        bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
        net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["model_state_dict"])
        net.eval()
        net.unc_ref = m["unc_ref"]
        sc = dict(np.load(os.path.join(d, "scalers.npz")))

        t_fwd = _cronometrar(lambda: C.predict(net, bte, t, sc, device, batch))
        c, lo, up = C.predict(net, bte, t, sc, device, batch)
        c2, lo2, up2 = C.predict(net, bte, t, sc, device, 7)
        drift = (max(np.abs(c - c2).max(), np.abs(lo - lo2).max(), np.abs(up - up2).max())
                 / C.pred_scale(c, lo, up))
        if drift > 1e-4:
            raise C.ProtocolError(f"{d}: la prediccion cambia con el tamano de batch ({drift:.2e}).")

        t_sig = _cronometrar(lambda: branch_jacobian_sigma(net, bte, t, sc, device))
        _, s_norm = branch_jacobian_sigma(net, bte, t, sc, device)
        err = np.abs(yte - c)

        # La semibanda propia, si el modelo aprendio alguna (el determinista no)
        half = (up - lo) / 2
        own = half if np.max(half) > 1e-8 * max(np.abs(c).max(), 1e-30) else None

        res = {"model": kind, "lambda": lam, "seed": seed, "beta": a.beta,
               "n_cal": len(cal), "n_eval": len(ev), "split_seed": SPLIT_SEED,
               "mse": float((err ** 2).mean()), "params": int(sum(p.numel() for p in net.parameters())),
               "ms_por_muestra": {"forward": 1e3 * t_fwd / len(bte),
                                  "sigma": 1e3 * t_sig / len(bte)},
               "shapes": {}}
        for name, h in shapes(s_norm, sc["Y_std"], a.beta, own).items():
            q = float(np.quantile((err / h)[cal], 1 - C.ALPHA))
            lo_q, up_q = c - q * h, c + q * h
            r = C.metrics(yte[ev], c[ev], lo_q[ev], up_q[ev])
            if r["nonfinite"]:
                raise C.ProtocolError(f"{d}/{name}: metricas conformales no finitas")
            r["q"] = q
            # Una forma constante no tiene rango que correlacionar: se omite en
            # vez de escribir NaN, que ademas no es JSON valido.
            if np.ptp(h[ev]) > 0:          # numpy 2: ndarray.ptp ya no existe
                r["rho_err"] = float(spearmanr(h[ev].ravel(), err[ev].ravel())[0])
                if jac_ref is not None:
                    r["rho_sens"] = float(spearmanr(h[ev].ravel(), jac_ref[ev].ravel())[0])
            res["shapes"][name] = r
        res.update({"commit_eval": commit, "test_md5": man["files"]["test"]["md5"],
                    "batch_invariance_rel_drift": float(drift)})
        C.write_json(out, res)
        rows.append(res)
        base = res["shapes"]["const_phys"]["mpiw"]
        print(f"  {os.path.basename(d):26s} " + "  ".join(
            f"{n}: {v['mpiw'] / base:5.3f}x PICP {v['picp']:5.2f}" for n, v in res["shapes"].items()),
            flush=True)

    print("\n| Modelo | l | Semilla | Forma | q | PICP (%) | MPIW@90% | vs const | IS | rhoErr | rhoSens |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        base = r["shapes"]["const_phys"]["mpiw"]
        for n, v in r["shapes"].items():
            sens = f"{v['rho_sens']:.3f}" if "rho_sens" in v else "-"
            rerr = f"{v['rho_err']:.3f}" if "rho_err" in v else "-"
            print(f"| {r['model']} | {r['lambda']:g} | {r['seed']} | {n} | {v['q']:.4g} | "
                  f"{v['picp']:.2f} | {v['mpiw']:.4e} | {v['mpiw'] / base:.3f} | "
                  f"{v['interval_score']:.4e} | {rerr} | {sens} |")


if __name__ == "__main__":
    main()
