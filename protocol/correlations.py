"""
Benchmark completo en TEST de las corridas completadas: metricas de
calidad + correlaciones de rango del paper.

    python protocol/correlations.py <benchmark> [--smoke] [--lam L] [--seed S]
    python protocol/correlations.py <benchmark> --ood <nombre>     (conjunto OOD de la config)

  rho_Err  : Spearman entre el ancho del intervalo y |error| (todos los puntos).
  rho_Sens : Spearman entre el ancho y |du/dparametro| de referencia
             (jacobian_reference del test; solo si el test la trae).

Misma definicion que results/abstract_benchmark.py (compute_uq_metrics): las
correlaciones se calculan sobre todos los puntos muestra x nodo juntos.

Mismas garantias que evaluate.py: test verificado por md5, sigma/arquitectura/
scalers/transformacion/referencia tomados de la metadata de cada corrida, y
prediccion invariante al tamano de batch. Escribe test_correlations.json en
cada corrida (nunca sobrescribe). Las corridas no completadas se omiten y se
informan: estas cifras son por corrida, la tabla oficial la hace aggregate.py.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--lam", type=float, help="Solo corridas con este lambda")
    ap.add_argument("--seed", type=int, help="Solo corridas con esta semilla")
    ap.add_argument("--ood", help="Evaluar sobre el conjunto OOD de la config en vez del test")
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    cfg = C.load_config(a.benchmark, a.smoke)
    man = C.load_manifest(cfg)
    if a.ood:
        data_path = C.ood_path(cfg, a.ood)
        if not os.path.exists(data_path):
            raise C.ProtocolError(f"Falta el conjunto OOD: {data_path}")
        data_md5, out_name, n_eval = C.md5_file(data_path), f"ood_{a.ood}.json", cfg["ood"][a.ood]["n"]
    else:
        C.require_data_matches_manifest(cfg, man, ("test",))
        data_path, data_md5 = C.split_path(cfg, "test"), man["files"]["test"]["md5"]
        out_name, n_eval = "test_correlations.json", cfg["data"]["splits"]["test"]["n"]
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    ap_ = cfg["data"]["append_params"]
    n_te = n_eval
    bte_raw, t, yte = C.load_split(data_path, ap_, n_te)
    with h5py.File(data_path, "r") as f:
        jac = f["jacobian_reference"][:n_te] if "jacobian_reference" in f else None
    if jac is not None:
        jac = np.abs(jac.sum(axis=2) if jac.ndim == 3 else jac)
    batch = cfg["train"]["val_infer_batch"]

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
        out = os.path.join(d, out_name)
        C.require_absent(out)

        bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
        net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["model_state_dict"])
        net.eval()
        net.unc_ref = m["unc_ref"]
        sc = dict(np.load(os.path.join(d, "scalers.npz")))

        c, lo, up = C.predict(net, bte, t, sc, device, batch)
        c2, lo2, up2 = C.predict(net, bte, t, sc, device, 7)
        drift = (max(np.abs(lo - lo2).max(), np.abs(up - up2).max(), np.abs(c - c2).max())
                 / C.pred_scale(c, lo, up))
        if drift > 1e-4:
            raise C.ProtocolError(f"{d}: la prediccion cambia con el tamano de batch ({drift:.2e}).")

        res = C.metrics(yte, c, lo, up)
        w = (up - lo).ravel()
        err = np.abs(yte - c).ravel()
        res["rho_err_spearman"] = float(spearmanr(w, err)[0])
        res["rho_err_pearson"] = float(pearsonr(w, err)[0])
        if jac is not None:
            res["rho_sens_spearman"] = float(spearmanr(w, jac.ravel())[0])
            res["rho_sens_pearson"] = float(pearsonr(w, jac.ravel())[0])
        res["params"] = int(sum(p.numel() for p in net.parameters()))
        res.update({"commit_eval": commit, "dataset": os.path.basename(data_path), "data_md5": data_md5,
                    "batch_invariance_rel_drift": float(drift)})
        C.write_json(out, res)
        rows.append((kind, lam, seed, res))

    print(f"\n| Modelo | λ | Semilla | Params | MSE | PICP (%) | MPIW | NLL | IS | ρErr | ρSens |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for kind, lam, seed, r in rows:
        sens = f"{r['rho_sens_spearman']:.3f}" if "rho_sens_spearman" in r else "–"
        print(f"| {kind} | {lam:g} | {seed} | {r['params'] / 1e6:.2f}M | {r['mse']:.3e} | {r['picp']:.2f} | "
              f"{r['mpiw']:.3e} | {r['nll']:.2f} | {r['interval_score']:.3e} | {r['rho_err_spearman']:.3f} | {sens} |")


if __name__ == "__main__":
    main()
