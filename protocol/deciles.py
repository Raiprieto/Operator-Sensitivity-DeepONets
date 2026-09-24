"""
Analisis por deciles de error en TEST: Jacobian vs Constant-width pinball
para cada lambda del barrido, con las figuras del paper.

    python protocol/deciles.py <benchmark> [--seed S] [--smoke]

Reutiliza results/plot_deciles_comparison_sweep.py: el calculo por decil
(decile_stats) y las figuras (plot_side_by_side_deciles), para que el
formato sea identico al de las figuras del paper. Lo unico que cambia es de
donde salen las predicciones: de las corridas del protocolo, con sigma,
arquitectura, scalers, transformacion de entrada y referencia de
incertidumbre tomados de la metadata de cada corrida (prediccion
determinista por muestra).

Exige que todas las corridas del barrido con la semilla pedida esten
completadas. Escribe en runs/<protocol>/<benchmark>/deciles_seed<S>/ y nunca
sobrescribe.
"""

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(C.REPO, "results"))
from plot_deciles_comparison_sweep import decile_stats, plot_side_by_side_deciles  # noqa: E402

EQ_NAMES = {"ns2d": "2D Compressible Navier-Stokes", "darcy": "2D Steady-State Darcy Flow",
            "burgers": "1D Viscous Burgers Equation", "biharmonic": "2D Biharmonic Equation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", default="vanilla",
                    help="Modelo del panel derecho (el izquierdo es jacobian si existe)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    cfg = C.load_config(a.benchmark, a.smoke)
    man = C.load_manifest(cfg)
    C.require_data_matches_manifest(cfg, man, ("test",))
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    out_dir = os.path.join(C.runs_dir(cfg), f"deciles_seed{a.seed}")
    C.require_absent(out_dir)

    runs = [r for r in C.plan(cfg) if r[2] == a.seed]
    if not runs:
        raise C.ProtocolError(f"La semilla {a.seed} no esta en el barrido")
    metas = {}
    for kind, lam, seed in runs:
        d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
        mp = os.path.join(d, "metadata.json")
        if not os.path.exists(mp):
            raise C.ProtocolError(f"Falta la corrida {d}")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m["status"] != "completed":
            raise C.ProtocolError(f"{d} no esta completada (status={m['status']})")
        if m["data_md5"] != {s: man["files"][s]["md5"] for s in C.SPLITS}:
            raise C.ProtocolError(f"{d} se entreno con otros datos que los del manifiesto.")
        metas[(kind, lam)] = (d, m)

    ap_ = cfg["data"]["append_params"]
    bte_raw, t, yte = C.load_split(C.split_path(cfg, "test"), ap_, cfg["data"]["splits"]["test"]["n"])
    batch = cfg["train"]["val_infer_batch"]

    results = collections.defaultdict(dict)
    for (kind, lam), (d, m) in sorted(metas.items()):
        bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
        net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
        net.load_state_dict(torch.load(os.path.join(d, "best.pt"), map_location=device)["model_state_dict"])
        net.eval()
        net.unc_ref = m["unc_ref"]
        sc = dict(np.load(os.path.join(d, "scalers.npz")))
        c, lo, up = C.predict(net, bte, t, sc, device, batch)
        key = int(lam) if float(lam).is_integer() else lam
        results[kind][key] = decile_stats(yte, c, lo, up)

    os.makedirs(out_dir)
    eq = EQ_NAMES.get(cfg.get("_data_benchmark", cfg["benchmark"]), cfg["benchmark"])
    izq = "jacobian" if results.get("jacobian") else sorted(results)[0]
    der = a.compare if results.get(a.compare) else next((k for k in sorted(results) if k != izq), izq)
    print(f"Figuras: panel izquierdo = {izq}, panel derecho = {der}")
    plot_side_by_side_deciles(results[izq], results[der], eq, out_dir)
    C.write_json(os.path.join(out_dir, "deciles.json"),
                 {"benchmark": cfg["benchmark"], "seed": a.seed, "commit": commit,
                  "test_md5": man["files"]["test"]["md5"], "results": dict(results)})

    for kind in sorted(results):
        print(f"\n### {kind}: PICP (%) por decil de error (D1 = menor error, D10 = mayor)")
        print("| λ | " + " | ".join(f"D{i}" for i in range(1, 11)) + " | Global |")
        print("|---" * 12 + "|")
        for lam, r in results[kind].items():
            print(f"| {lam} | " + " | ".join(f"{x:.1f}" for x in r["coverage_mean"]) + f" | {r['coverage_global']:.1f} |")
        print(f"\n### {kind}: MPIW por decil de error")
        print("| λ | " + " | ".join(f"D{i}" for i in range(1, 11)) + " | D10/D1 |")
        print("|---" * 12 + "|")
        for lam, r in results[kind].items():
            w = r["mpiw_mean"]
            print(f"| {lam} | " + " | ".join(f"{x:.3f}" for x in w) + f" | {w[-1] / w[0]:.2f} |")
    print(f"\nFiguras y datos en {out_dir}")


if __name__ == "__main__":
    main()
