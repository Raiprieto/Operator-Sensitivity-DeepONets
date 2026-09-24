"""
biharmonic_scaled_test.py
=========================
Genera un conjunto de test de la ecuacion biarmonica con muestras NUEVAS,
disjuntas del entrenamiento, sin volver a correr el solver FEM.

Por que es exacto
-----------------
El dataset de fair-sciml (data/biharmonic_equation.h5) es una familia de un
solo parametro: la carga es f(x) = c * g(x) con una forma g fija, y como la
PDE es lineal, la solucion es u(x; c) = c * u_base(x). Verificado sobre las
3010 simulaciones: f/c y u/c coinciden entre todas ellas hasta ~1e-14, y la
matriz de cargas tiene rango 1.

Por lo tanto, para un c nuevo, (c * g, c * u_base) es exactamente la solucion
FEM que devolveria DOLFINx con esa malla, hasta precision de maquina.

Distribucion
------------
En el train, c ~ U(1, 5) (histograma plano en [1, 5]). Aqui se muestrea c de
la misma distribucion con una semilla propia, y se descarta cualquier valor
que coincida con un c del train.

Formato de salida: identico a data/biharmonic_test.h5 (el que produce
generate_fem_jacobians.py), para que los scripts de evaluacion no cambien.

USO:
    python src/data_generation/biharmonic_scaled_test.py \
        --train_path  data/biharmonic_equation.h5 \
        --output_path data/biharmonic_test_clean.h5 \
        --num_samples 3010 --seed 20260921
"""

import argparse
import os

import h5py
import numpy as np


def leer_train(path):
    F, U, C, coords = [], [], [], None
    with h5py.File(path, "r") as f:
        for s in f:
            for sim in f[s]:
                g = f[s][sim]
                F.append(g["field_input_f"][:])
                U.append(g["values"][:])
                C.append(float(g.attrs["parameter_coefficient"]))
                if coords is None:
                    coords = g["coordinates"][:]
    return np.array(F), np.array(U), np.array(C), coords


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train_path", default="data/biharmonic_equation.h5")
    p.add_argument("--output_path", default="data/biharmonic_test_clean.h5")
    p.add_argument("--num_samples", type=int, default=3010)
    p.add_argument("--c_min", type=float, default=1.0)
    p.add_argument("--c_max", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=20260921)
    args = p.parse_args()

    F, U, C, coords = leer_train(args.train_path)

    # Formas base (c = 1). Se promedian por robustez; son identicas a ~1e-14.
    g = (F / C[:, None]).mean(axis=0)
    u_base = (U / C[:, None]).mean(axis=0)

    # Verificar la hipotesis de linealidad antes de confiar en ella
    err_f = np.abs(F - C[:, None] * g).max() / np.abs(F).max()
    err_u = np.abs(U - C[:, None] * u_base).max() / np.abs(U).max()
    print(f"Hipotesis f = c*g      : error relativo max {err_f:.2e}")
    print(f"Hipotesis u = c*u_base : error relativo max {err_u:.2e}")
    if max(err_f, err_u) > 1e-10:
        raise SystemExit("ERROR: los datos no son lineales en c; este metodo no aplica.")

    # Muestrear c nuevos de la misma distribucion, disjuntos del train
    rng = np.random.default_rng(args.seed)
    train_c = set(np.round(C, 12))
    c_new = []
    while len(c_new) < args.num_samples:
        c = rng.uniform(args.c_min, args.c_max)
        if np.round(c, 12) not in train_c:
            c_new.append(c)
    c_new = np.array(c_new)

    branch = c_new[:, None] * g[None, :]
    targets = c_new[:, None] * u_base[None, :]
    # Mismo jacobiano que generate_fem_jacobians.py: du/dc = u/c = u_base
    jac = np.repeat(u_base[None, :], args.num_samples, axis=0)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with h5py.File(args.output_path, "w") as f:
        f.create_dataset("branch_inputs", data=branch, compression="gzip")
        f.create_dataset("trunk_inputs", data=coords[:, :2], compression="gzip")
        f.create_dataset("targets", data=targets, compression="gzip")
        f.create_dataset("jacobian_reference", data=jac, compression="gzip")
        f.create_dataset("params_coefficient", data=c_new, compression="gzip")
        f.attrs["equation"] = "biharmonic"
        f.attrs["derivation"] = "analytical_linearity"
        f.attrs["n_nodes"] = branch.shape[1]
        f.attrs["n_samples"] = args.num_samples
        f.attrs["seed"] = args.seed
        f.attrs["description"] = (
            "Held-out biharmonic test set. New coefficients c ~ U(1,5) with an "
            "independent seed, disjoint from training. Exact FEM solutions "
            "obtained by linearity (u = c * u_base) from the fair-sciml data.")

    print(f"Escrito {args.output_path}: {args.num_samples} muestras, "
          f"c en [{c_new.min():.4f}, {c_new.max():.4f}]")


if __name__ == "__main__":
    main()
