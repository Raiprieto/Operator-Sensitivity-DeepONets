"""
fair_eval.py -- uniform evaluation of every band-producing DeepONet variant on a held-out test set.

For each model stem (e.g. modelos/burgers_fair/jac) it reads
    <stem>_args.json     (CLI args used for training -> rebuild the exact network)
    <stem>_scalers.npz   (z-score parameters -> back to physical units)
    <stem>-<iter>.pt     (checkpoint; the latest one is used unless --iter is given)
and computes, in PHYSICAL units, the metrics used in the paper:
    MSE, rel-L2, PICP (90% nominal), MPIW, Gaussian proxy-NLL (std = width / (2*1.645)),
    Spearman rho(width, |error|) pointwise, Spearman rho(width, |J_ref|) pointwise when the
    test file has `jacobian_reference`, per-sample-MSE decile coverage / MPIW (paper's decile
    figures), number of trainable parameters, and inference latency per sample.

Usage:
    python results/fair_eval.py --test data/burgers_test_500.h5 \
        --models det=modelos/burgers_fair/det cw=modelos/burgers_fair/cw \
                 qx=modelos/burgers_fair/qx qux=modelos/burgers_fair/qux jac=modelos/burgers_fair/jac \
        --out results/fair/burgers
"""
import argparse, glob, json, os, sys, time
import numpy as np
import h5py
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
# results/utils.py shadows src/model_training/utils.py -> make the trainer's dir win and drop results/
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, os.path.join(HERE, "..", "deepxde-extensions"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "model_training"))
os.environ.setdefault("DDE_BACKEND", "pytorch")
import deepxde as dde  # noqa: E402
dde.config.set_default_float("float32")
from deeponet_training import crear_red, read_data  # noqa: E402


def latest_ckpt(stem, it=None):
    if it is not None:
        return f"{stem}-{it}.pt"
    cands = glob.glob(f"{stem}-*.pt")
    if not cands:
        raise FileNotFoundError(f"no checkpoint for {stem}")
    return max(cands, key=lambda p: int(p.rsplit("-", 1)[1][:-3]))


def load_model(stem, device, it=None):
    with open(f"{stem}_args.json") as fh:
        args = argparse.Namespace(**json.load(fh))
    net = crear_red(args)
    ck = torch.load(latest_ckpt(stem, it), map_location=device)
    state = ck["model_state_dict"] if "model_state_dict" in ck else ck
    res = net.load_state_dict(state, strict=False)   # older checkpoints lack the EMA-scale buffers
    if res.missing_keys or res.unexpected_keys:
        print(f"  [load_model] {os.path.basename(stem)}: missing={res.missing_keys} unexpected={res.unexpected_keys}")
    net.to(device).eval()
    sc = np.load(f"{stem}_scalers.npz")
    scalers = {k: sc[k] for k in sc.files}
    nparams = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net, args, scalers, nparams


@torch.no_grad()
def _fwd_no_grad(net, u, x):
    return net((u, x))


def predict_physical(net, args, scalers, branch_raw, trunk_raw, device, bs=64):
    """Returns y, y_low, y_up in physical units, shape [n, N]. Bands are None for the deterministic model."""
    ub = (branch_raw - scalers["branch_mean"]) / scalers["branch_std"]
    xt = (trunk_raw - scalers["trunk_mean"]) / scalers["trunk_std"]
    xt_t = torch.tensor(xt, dtype=torch.float32, device=device)
    needs_grad = bool(getattr(args, "use_jacobian", False))
    ys, lows, ups = [], [], []
    # warm-up (CUDA context / kernel autotuning) so the first model measured is not penalised
    u0 = torch.tensor(ub[:min(bs, len(ub))], dtype=torch.float32, device=device)
    if needs_grad:
        u0.requires_grad_(True); net((u0, xt_t))
    else:
        _fwd_no_grad(net, u0, xt_t)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(ub), bs):
        u = torch.tensor(ub[i:i + bs], dtype=torch.float32, device=device)
        if needs_grad:
            u.requires_grad_(True)
            out = net((u, xt_t))
        else:
            out = _fwd_no_grad(net, u, xt_t)
        out = out.detach().cpu().numpy()
        N = trunk_raw.shape[0]
        if out.shape[-1] == 3 * N:
            ys.append(out[:, :N]); lows.append(out[:, N:2 * N]); ups.append(out[:, 2 * N:])
        else:
            ys.append(out)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    lat_ms = (time.perf_counter() - t0) / len(ub) * 1e3
    Ym, Ys = scalers["Y_mean"], scalers["Y_std"]
    y = np.concatenate(ys) * Ys + Ym
    if lows:
        lo = np.concatenate(lows) * Ys + Ym
        up = np.concatenate(ups) * Ys + Ym
    else:
        lo = up = None
    return y, lo, up, lat_ms


def conformal_q(y, lo, up, Yt, alpha=0.10):
    """Split-conformal multiplier from a calibration set: (1-alpha) finite-sample quantile of |err|/half-width."""
    hl = np.maximum(y - lo, 1e-12); hu = np.maximum(up - y, 1e-12)
    err = Yt - y
    sc = np.where(err < 0, -err / hl, err / hu).ravel()
    k = int(np.ceil((sc.size + 1) * (1 - alpha))) / sc.size
    return float(np.quantile(sc, min(k, 1.0)))


def apply_q(y, lo, up, q):
    return y - q * np.maximum(y - lo, 1e-12), y + q * np.maximum(up - y, 1e-12)


def conformal_rescale(y, lo, up, Yt, alpha=0.10, calib_frac=0.5):
    """Split-conformal rescaling of the bands: score s = max((y-lo_dist), (up_dist)) normalised by the
    half-widths on the first `calib_frac` of the samples; the (1-alpha) finite-sample quantile q rescales
    both half-widths on the remaining samples. Post-hoc, identical for every model -> equal-coverage
    comparison of sharpness (MPIW), NLL and decile behaviour. Returns (y, lo, up, Yt) on the eval split + q."""
    n = len(y); nc = int(n * calib_frac)
    hl = np.maximum(y - lo, 1e-12); hu = np.maximum(up - y, 1e-12)
    err = Yt - y
    s = np.where(err < 0, -err / hl, err / hu)          # >1  <=> outside the band
    sc = s[:nc].ravel()
    k = int(np.ceil((sc.size + 1) * (1 - alpha))) / sc.size
    q = float(np.quantile(sc, min(k, 1.0)))
    ye, loe, upe, Yte = y[nc:], y[nc:] - q * hl[nc:], y[nc:] + q * hu[nc:], Yt[nc:]
    return ye, loe, upe, Yte, q


def metrics(y, lo, up, Yt, Jref=None, n_dec=10, rng=None):
    m = {}
    err = Yt - y
    m["mse"] = float(np.mean(err ** 2))
    m["rel_l2"] = float(np.linalg.norm(err) / np.linalg.norm(Yt))
    if lo is None:
        return m
    w = up - lo
    inside = (Yt >= lo) & (Yt <= up)
    m["picp"] = float(inside.mean() * 100)
    m["mpiw"] = float(w.mean())
    std = w / (2 * 1.644853)
    var = std ** 2 + 1e-8
    m["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * err ** 2 / var))
    m["crossing_frac"] = float((w < 0).mean())
    # pointwise rank correlations on a fixed random subsample (2e5 points) for speed
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(w.size, size=min(200_000, w.size), replace=False)
    m["rho_err"] = float(spearmanr(w.ravel()[idx], np.abs(err).ravel()[idx]).statistic)
    if Jref is not None:
        m["rho_sens"] = float(spearmanr(w.ravel()[idx], np.abs(Jref).ravel()[idx]).statistic)
    # per-sample-MSE deciles (paper's decile figures)
    mse_s = np.mean(err ** 2, axis=1)
    cov_s = inside.mean(axis=1) * 100
    mpiw_s = w.mean(axis=1)
    order = np.argsort(mse_s)
    m["decile_picp"] = [float(np.mean(c)) for c in np.array_split(cov_s[order], n_dec)]
    m["decile_mpiw"] = [float(np.mean(c)) for c in np.array_split(mpiw_s[order], n_dec)]
    m["decile_mse"] = [float(np.mean(c)) for c in np.array_split(mse_s[order], n_dec)]
    # per-sample Spearman(width, |err|) averaged over samples (spatial alignment inside each field)
    per = [spearmanr(w[i], np.abs(err[i])).statistic for i in range(0, len(w), max(1, len(w) // 200))]
    m["rho_err_within_sample"] = float(np.nanmean(per))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=stem pairs")
    ap.add_argument("--out", required=True, help="output prefix (json + md + png)")
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--calib_test", default=None, help="h5 file used ONLY to calibrate the conformal multiplier q (e.g. in-distribution test); --test is then fully used for evaluation (OOD protocol)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ns = argparse.Namespace(data_path=a.test, num_branch_sensors=0)
    branch, trunk, Yt = read_data(ns)
    Jref = None
    with h5py.File(a.test) as f:
        if "jacobian_reference" in f:
            Jref = f["jacobian_reference"][:]
    if a.n_test:
        branch, Yt = branch[:a.n_test], Yt[:a.n_test]
        if Jref is not None:
            Jref = Jref[:a.n_test]
    branch = branch.astype(np.float32); trunk = trunk.astype(np.float32); Yt = Yt.astype(np.float32)

    if a.calib_test:
        nsc = argparse.Namespace(data_path=a.calib_test, num_branch_sensors=0)
        cb, ct, cY = read_data(nsc)
        cb, ct, cY = cb.astype(np.float32), ct.astype(np.float32), cY.astype(np.float32)

    results = {}
    for pair in a.models:
        name, stem = pair.split("=", 1)
        net, args, scalers, nparams = load_model(stem, a.device, a.iter)
        y, lo, up, lat = predict_physical(net, args, scalers, branch, trunk, a.device)
        m = metrics(y, lo, up, Yt, Jref)
        if lo is not None and a.calib_test:
            cy, clo, cup, _ = predict_physical(net, args, scalers, cb, ct, a.device)
            q = conformal_q(cy, clo, cup, cY)
            loe, upe = apply_q(y, lo, up, q)
            mc = metrics(y, loe, upe, Yt, Jref)
            m["conf_q"] = q
            for k in ("picp", "mpiw", "nll", "rho_err", "decile_picp", "decile_mpiw"):
                m["conf_" + k] = mc[k]
        elif lo is not None:
            ye, loe, upe, Yte = conformal_rescale(y, lo, up, Yt)[:4]
            q = conformal_rescale(y, lo, up, Yt)[4]
            Je = Jref[len(Jref) - len(Yte):] if Jref is not None else None
            mc = metrics(ye, loe, upe, Yte, Je)
            m["conf_q"] = q
            for k in ("picp", "mpiw", "nll", "rho_err", "decile_picp", "decile_mpiw"):
                m["conf_" + k] = mc[k]
        m["n_params"] = int(nparams)
        m["latency_ms_per_sample"] = float(lat)
        m["ckpt"] = latest_ckpt(stem, a.iter)
        m["pinball_lambda"] = getattr(args, "pinball_lambda", None)
        results[name] = m
        print(f"[{name}] " + ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)))
        del net; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(results, fh, indent=2)

    # markdown table
    cols = ["n_params", "mse", "rel_l2", "picp", "mpiw", "nll", "rho_err", "rho_err_within_sample", "rho_sens", "latency_ms_per_sample",
            "conf_q", "conf_picp", "conf_mpiw", "conf_nll"]
    lines = ["| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, m in results.items():
        row = []
        for c in cols:
            v = m.get(c)
            row.append("" if v is None else (f"{v:d}" if isinstance(v, int) else f"{v:.4g}"))
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    dec = ["", "Decile PICP (per-sample MSE deciles, D1 easiest -> D10 hardest):", "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        if "decile_picp" in m:
            dec.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in m["decile_picp"]) + " |")
    dec += ["", "Decile MPIW:", "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        if "decile_mpiw" in m:
            dec.append(f"| {name} | " + " | ".join(f"{v:.3g}" for v in m["decile_mpiw"]) + " |")
    dec += ["", ("Split-conformal rescaled (q calibrated on %s, evaluated on the FULL --test file) -- decile PICP:" % os.path.basename(a.calib_test)) if a.calib_test else "Split-conformal rescaled (calibrated on first half of test, evaluated on second half) -- decile PICP:",
            "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        if "conf_decile_picp" in m:
            dec.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in m["conf_decile_picp"]) + " |")
    dec += ["", "Split-conformal rescaled -- decile MPIW:", "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        if "conf_decile_mpiw" in m:
            dec.append(f"| {name} | " + " | ".join(f"{v:.3g}" for v in m["conf_decile_mpiw"]) + " |")
    with open(a.out + ".md", "w") as fh:
        fh.write("\n".join(lines + dec) + "\n")
    print("\n".join(lines + dec))

    # figure: decile coverage + decile MPIW
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, m in results.items():
            if "decile_picp" in m:
                ax[0].plot(range(1, 11), m["decile_picp"], marker="o", label=name)
                ax[1].semilogy(range(1, 11), m["decile_mpiw"], marker="o", label=name)
        ax[0].axhline(90, ls="--", c="k", lw=1); ax[0].set_ylabel("PICP (%)"); ax[0].set_xlabel("per-sample MSE decile")
        ax[1].set_ylabel("MPIW"); ax[1].set_xlabel("per-sample MSE decile"); ax[0].legend()
        fig.suptitle(os.path.basename(a.out)); fig.tight_layout(); fig.savefig(a.out + "_deciles.png", dpi=130)
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


if __name__ == "__main__":
    main()