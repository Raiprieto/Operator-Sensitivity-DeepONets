"""
check_leakage.py
================
Verifica que ninguna muestra de un conjunto de test aparezca en el de train.

Compara cada fila de test contra todas las de train por hash de sus inputs
(branch_inputs y, si existen, params_coefficient). Acepta archivos planos y
el formato jerarquico de fair-sciml (sesiones -> simulaciones).

Sale con codigo 1 si encuentra solapamiento, para que un sbatch con `set -e`
aborte antes de evaluar sobre datos contaminados.

USO:
    python src/data_generation/check_leakage.py \
        --train data/ns2d_state_train.h5 --test data/ns2d_state_test_clean.h5
"""

import argparse
import hashlib
import sys

import h5py
import numpy as np


def cargar_inputs(path):
    """Devuelve (branch, params); params es None si el archivo no los guarda."""
    with h5py.File(path, "r") as f:
        if "branch_inputs" in f:
            x = f["branch_inputs"][:]
            p = f["params_coefficient"][:] if "params_coefficient" in f else None
            return x.reshape(len(x), -1), (None if p is None else p.reshape(len(p), -1))
        # fair-sciml: la carga ya contiene el coeficiente (f = c * g)
        filas = [f[s][sim]["field_input_f"][:] for s in f for sim in f[s]]
        return np.array(filas), None


def hashes(x, decimales=None):
    if decimales is not None:
        x = np.round(x.astype(np.float64), decimales)
    x = np.ascontiguousarray(x)
    return [hashlib.md5(fila.tobytes()).hexdigest() for fila in x]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    args = p.parse_args()

    (tr, ptr), (te, pte) = cargar_inputs(args.train), cargar_inputs(args.test)
    # Los parametros solo entran en la comparacion si ambos archivos los tienen
    if ptr is not None and pte is not None:
        tr = np.concatenate([tr, ptr.astype(tr.dtype)], axis=1)
        te = np.concatenate([te, pte.astype(te.dtype)], axis=1)
    if tr.shape[1] != te.shape[1]:
        sys.exit(f"ERROR: dimensiones incompatibles {tr.shape} vs {te.shape}")

    peor = 0
    for etiqueta, dec in (("exacto", None), ("redondeo 1e-6", 6)):
        ht = set(hashes(tr, dec))
        n = sum(h in ht for h in hashes(te, dec))
        peor = max(peor, n)
        print(f"  [{etiqueta:14s}] {n:6d} de {len(te)} muestras de test estan en train")

    # Distancia minima test->train, para detectar casi-duplicados
    idx = np.random.default_rng(0).choice(len(te), size=min(200, len(te)), replace=False)
    trf = tr.astype(np.float64)
    escala = np.linalg.norm(trf, axis=1).mean()
    dmin = min(np.linalg.norm(trf - te[i].astype(np.float64), axis=1).min() for i in idx)
    print(f"  distancia minima test->train (muestra de {len(idx)}): "
          f"{dmin:.3e}  (relativa a la norma media: {dmin / escala:.2e})")

    if peor:
        print(f"LEAKAGE: {args.test} comparte muestras con {args.train}")
        sys.exit(1)
    print(f"OK: {args.test} es disjunto de {args.train}")


if __name__ == "__main__":
    main()
