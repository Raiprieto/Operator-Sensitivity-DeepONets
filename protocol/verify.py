"""
Verifica los tres splits de un benchmark y escribe su MANIFEST.json.

    python protocol/verify.py <benchmark> [--smoke]

El manifiesto es el certificado que protocol/train.py exige antes de entrenar.
Solo se escribe si TODAS estas condiciones se cumplen; si no, aborta:

  1. Cada split existe y tiene exactamente las n muestras de la configuracion.
  2. Todos comparten la misma grilla (trunk) y las dimensiones esperadas por el modelo.
  3. Ninguna muestra se repite entre train/val, train/test ni val/test
     (comparacion exacta y con redondeo a 1e-6).
  4. Val y test son indistinguibles del train en distribucion (KS sobre
     estadisticos por muestra; aborta si p < 1e-4).

Una vez escrito, el manifiesto congela los datos: el entrenamiento compara el
md5 de cada archivo contra el registrado y aborta ante cualquier diferencia.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(__file__))
from common import (SPLITS, ProtocolError, load_config, load_split, manifest_path,  # noqa: E402
                    md5_file, require_absent, require_clean_code, require_slurm, split_path,
                    transform_branch, write_json)
from check_leakage import cargar_inputs, hashes  # noqa: E402  (src/data_generation)

KS_MIN_P = 1e-4


def overlap(pa, pb):
    (a, qa), (b, qb) = cargar_inputs(pa), cargar_inputs(pb)
    if qa is not None and qb is not None:
        a = np.concatenate([a, qa.astype(a.dtype)], 1)
        b = np.concatenate([b, qb.astype(b.dtype)], 1)
    if a.shape[1] != b.shape[1]:
        raise ProtocolError(f"Dimensiones incompatibles al comparar {pa} y {pb}: {a.shape} vs {b.shape}")
    res = {}
    for tag, dec in (("exact", None), ("round_1e-6", 6)):
        ha = set(hashes(a, dec))
        res[tag] = int(sum(h in ha for h in hashes(b, dec)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    require_slurm()
    commit = require_clean_code()
    cfg = load_config(a.benchmark, a.smoke)
    require_absent(manifest_path(cfg))
    ap_ = cfg["data"]["append_params"]

    data, files = {}, {}
    for s in SPLITS:
        p = split_path(cfg, s)
        if not os.path.exists(p):
            raise ProtocolError(f"Falta el split '{s}': {p}")
        n = cfg["data"]["splits"][s]["n"]
        b, t, y = load_split(p, ap_, n)
        if len(b) != n:
            raise ProtocolError(f"'{s}' tiene {len(b)} muestras y la configuracion exige {n}")
        data[s] = (b, t, y)
        files[s] = {"path": p, "md5": md5_file(p), "n": n}
        print(f"  {s:5s} {p}  n={n}  branch={b.shape[1]}  nodos={y.shape[1]}")

    bt, tt, yt = data["train"]
    # Con input_transform, lo que debe calzar con el modelo es la entrada transformada
    transform_branch(bt, cfg, cfg["model"]["branch"][0])
    for s in ("val", "test"):
        b, t, y = data[s]
        if b.shape[1] != bt.shape[1] or y.shape[1] != yt.shape[1]:
            raise ProtocolError(f"Dimensiones de '{s}' distintas a las del train")
        if not np.array_equal(t, tt):
            raise ProtocolError(f"La grilla (trunk) de '{s}' no es la del train")

    print("\nSolapamiento entre splits:")
    leak = {}
    for x, z in (("train", "val"), ("train", "test"), ("val", "test")):
        leak[f"{x}-{z}"] = r = overlap(split_path(cfg, x), split_path(cfg, z))
        print(f"  {x:5s} vs {z:5s}: exacto={r['exact']}  redondeo_1e-6={r['round_1e-6']}")
    if any(v for r in leak.values() for v in r.values()):
        raise ProtocolError("Hay muestras compartidas entre splits (leakage).")

    print("\nDistribucion de val/test respecto del train (KS):")
    ks = {}
    feats = {"branch_mean": lambda b, y: b.mean(1), "branch_std": lambda b, y: b.std(1),
             "target_mean": lambda b, y: y.mean(1), "target_std": lambda b, y: y.std(1)}
    for s in ("val", "test"):
        b, _, y = data[s]
        for k, f in feats.items():
            r = ks_2samp(f(bt, yt), f(b, y))
            ks[f"{s}.{k}"] = {"stat": float(r.statistic), "p": float(r.pvalue)}
            flag = "  <-- DISTINTA" if r.pvalue < KS_MIN_P else ""
            print(f"  {s:4s} {k:12s} KS={r.statistic:.3f} p={r.pvalue:.3g}{flag}")
    bad = [k for k, v in ks.items() if v["p"] < KS_MIN_P]
    if bad:
        raise ProtocolError(f"Val/test no parecen venir de la distribucion del train: {bad}")

    write_json(manifest_path(cfg), {
        "benchmark": cfg["benchmark"], "smoke": cfg["_smoke"],
        "data_config_md5": cfg["_data_config_md5"], "config_md5": cfg["_config_md5"],
        "commit": commit, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "files": files, "leakage": leak, "leakage_ok": True, "ks": ks, "ks_min_p": KS_MIN_P})
    print(f"\nOK: manifiesto escrito en {manifest_path(cfg)}")


if __name__ == "__main__":
    main()
