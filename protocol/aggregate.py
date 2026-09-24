"""
Construye la tabla de resultados de un benchmark: media +- desviacion estandar
sobre TODAS las semillas declaradas.

    python protocol/aggregate.py <benchmark> [--smoke]

Reglas (si alguna no se cumple, aborta con codigo 3):
  - Todas las corridas del barrido deben estar completadas y evaluadas. No se
    reporta un subconjunto de semillas: eso permitiria quedarse con las buenas.
  - El lambda "seleccionado" de cada modelo se elige por el interval score
    medio en VALIDACION. El test no interviene en ninguna decision.

Escribe summary.json, summary.csv y summary.md en runs/<protocol>/<benchmark>/.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

KEYS = ("mse", "picp", "mpiw", "nll", "interval_score")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = C.load_config(a.benchmark, a.smoke)

    groups, missing = {}, []
    for kind, lam, seed in C.plan(cfg):
        d = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
        mp, tp = os.path.join(d, "metadata.json"), os.path.join(d, "test_metrics.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            missing.append(d)
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        with open(tp, encoding="utf-8") as f:
            t = json.load(f)
        if m["status"] != "completed":
            missing.append(f"{d} (status={m['status']})")
            continue
        groups.setdefault((kind, lam), []).append((seed, m, t))
    if missing:
        raise C.ProtocolError("Barrido incompleto; no se agrega un subconjunto:\n  " + "\n  ".join(missing))

    rows = []
    for (kind, lam), items in sorted(groups.items()):
        row = {"model": kind, "lambda": lam, "n_seeds": len(items),
               "seeds": [s for s, _, _ in items],
               "val_interval_score": float(np.mean([m["val"]["interval_score"] for _, m, _ in items]))}
        for k in KEYS:
            v = np.array([t["test"][k] for _, _, t in items], dtype=float)
            row[f"test_{k}_mean"] = float(v.mean())
            row[f"test_{k}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        leg = [t["test_legacy_batchmean"]["picp"] for _, _, t in items if t.get("test_legacy_batchmean")]
        row["test_picp_legacy_batchmean_mean"] = float(np.mean(leg)) if leg else None
        rows.append(row)

    selected = {}
    for kind in cfg["sweep"]["models"]:
        cand = [r for r in rows if r["model"] == kind]
        best = min(cand, key=lambda r: r["val_interval_score"])
        best["selected_by_val"] = True
        selected[kind] = best["lambda"]

    out = C.runs_dir(cfg)
    C.write_json(os.path.join(out, "summary.json"),
                 {"benchmark": cfg["benchmark"], "smoke": cfg["_smoke"],
                  "selection": "lambda con menor interval score medio en validacion",
                  "selected_lambda": selected, "rows": rows})
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["selected_by_val"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    fmt = lambda r, k, e=False: (f"{r[f'test_{k}_mean']:.3e} ± {r[f'test_{k}_std']:.1e}" if e  # noqa: E731
                                 else f"{r[f'test_{k}_mean']:.2f} ± {r[f'test_{k}_std']:.2f}")
    lines = [f"# {cfg['benchmark']} — test, media ± desv. estandar sobre semillas",
             "", "| Modelo | λ | Semillas | MSE | PICP (%) | MPIW | NLL | Interval score | Val IS | Sel. |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['lambda']:g} | {r['n_seeds']} | {fmt(r, 'mse', True)} | "
                     f"{fmt(r, 'picp')} | {fmt(r, 'mpiw', True)} | {fmt(r, 'nll')} | "
                     f"{fmt(r, 'interval_score', True)} | {r['val_interval_score']:.3e} | "
                     f"{'✓' if r.get('selected_by_val') else ''} |")
    lines += ["", "Sel. = λ elegido por menor interval score medio en validación (el test no interviene).",
              "Métricas con referencia de incertidumbre fija (intervalo determinista por muestra)."]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
