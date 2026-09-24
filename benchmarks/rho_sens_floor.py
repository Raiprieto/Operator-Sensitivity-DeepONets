"""rho_sens_floor.py -- the rho_Sens a constant width reaches through de-normalization alone.

Models are trained in z-scored coordinates and evaluated in physical ones, so a width that is
constant during training is returned multiplied by the per-node training standard deviation.
This reports the rank correlation of that field with the reference sensitivity, which is the
floor every rho_Sens in the paper sits above.

Usage:
    python results/rho_sens_floor.py \
        --pair burgers modelos/burgers_fair/jac_scalers.npz data/burgers_test_500.h5
"""
import argparse
import numpy as np
import h5py
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=3, action="append", metavar=("NAME", "SCALERS", "TEST"), required=True)
    p.add_argument("--n", type=int, default=200_000)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    for name, scalers, test in a.pair:
        ys = np.load(scalers)["Y_std"]
        with h5py.File(test, "r") as f:
            if "jacobian_reference" not in f:
                print(f"{name}: no jacobian_reference in {test}")
                continue
            J = np.abs(np.asarray(f["jacobian_reference"]))
        J = J.reshape(J.shape[0], -1)
        if J.shape[1] != ys.size:
            print(f"{name}: shape mismatch {J.shape} vs {ys.shape}")
            continue
        W = np.broadcast_to(ys, J.shape)
        idx = rng.choice(J.size, size=min(a.n, J.size), replace=False)
        rho = spearmanr(W.ravel()[idx], J.ravel()[idx]).statistic
        print(f"{name}: Y_std in [{ys.min():.3g}, {ys.max():.3g}]  rho_sens_floor = {rho:.4f}")


if __name__ == "__main__":
    main()