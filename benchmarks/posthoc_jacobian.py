"""
posthoc_jacobian.py -- "UQ for free" on an already-trained deterministic DeepONet.

Takes a plain DeepONetCartesianProd checkpoint (trained with MSE only), extracts the latent
branch Jacobian J_branch = dB/du with vmap(jacrev) (K reverse-mode VJPs), forms the Gram matrix
G = J J^T and the pointwise sensitivity  s(x) = sqrt(diag(T(x) G T(x)^T))  -- exactly the
O(K) construction of Jacobian-DeepONet, but with NO retraining and NO extra parameters.

The spatial shape  shape(x) = s(x) / mean_x s(x)  (optionally log1p as in the trained model) is then
turned into a prediction interval by a single scalar calibrated with split-conformal on the first
half of the test set:   y +/- q * shape(x),  q = (1-alpha) quantile of |err| / shape.
The second half is evaluated (PICP, MPIW, NLL, rho_err, rho_sens, decile PICP/MPIW).

Baselines computed with the same conformal recipe on the same split:
    const   : shape = 1 (split-conformal with a constant band = standard split-conformal)
    posthoc : Jacobian shape from the deterministic model (this script's contribution)
    posthoc_log1p : same with log1p(shape) (the transform used inside JacobianDeepONetSoftplus)
Optionally any trained band model (--bands name=stem) is added, rescaled with its own conformal q.

Usage:
    python results/posthoc_jacobian.py --test data/burgers_test_500.h5 --det modelos/burgers_fair/det \
        --bands qux=modelos/burgers_fair/qux jac=modelos/burgers_fair/jac --out results/fair/posthoc_burgers
"""
import argparse, json, os, sys
import numpy as np
import torch
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, HERE)  # for fair_eval
from fair_eval import load_model, predict_physical, metrics, conformal_rescale, conformal_q, apply_q, read_data  # noqa: E402


def jacobian_shape(net, scalers, branch_raw, trunk_raw, device, bs=32, clamp=50.0):
    """Pointwise latent-Jacobian sensitivity s(x) for a deterministic DeepONetCartesianProd, shape [n, N]."""
    from torch.func import vmap, jacrev
    ub = torch.tensor((branch_raw - scalers["branch_mean"]) / scalers["branch_std"], dtype=torch.float32, device=device)
    xt = torch.tensor((trunk_raw - scalers["trunk_mean"]) / scalers["trunk_std"], dtype=torch.float32, device=device)
    with torch.no_grad():
        T = net.activation_trunk(net.trunk(xt))                       # [N, K]
    def branch_single(u_s):
        return net.branch(u_s.unsqueeze(0)).squeeze(0)
    out = []
    for i in range(0, len(ub), bs):
        u = ub[i:i + bs]
        J = vmap(jacrev(branch_single))(u)                             # [b, K, N_in]
        J = torch.clamp(J, -clamp, clamp)
        G = torch.bmm(J, J.transpose(1, 2))                            # [b, K, K]
        TG = torch.einsum("nk,bkm->bnm", T, G)
        var = torch.einsum("bnm,nm->bn", TG, T).clamp_min(0)           # [b, N]
        out.append(torch.sqrt(var + 1e-12).detach().cpu().numpy())
    s = np.concatenate(out)                                            # normalised units, [n, N]
    return s


def mc_perturb_shape(net, scalers, branch_raw, trunk_raw, device, M=64, rel_sigma=0.05, bs=64, seed=0):
    """Post-hoc adaptive score by input-perturbation Monte Carlo through the SAME deterministic backbone:
    std over M forward passes with u + eps, eps ~ N(0, (rel_sigma*std(u))^2). Returns [n, N] in normalised output units."""
    rng = np.random.default_rng(seed)
    ub = (branch_raw - scalers["branch_mean"]) / scalers["branch_std"]
    xt = torch.tensor((trunk_raw - scalers["trunk_mean"]) / scalers["trunk_std"], dtype=torch.float32, device=device)
    N = trunk_raw.shape[0]
    out = []
    import time; t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(ub)):
            eps = rng.normal(0.0, rel_sigma, size=(M, ub.shape[1])).astype(np.float32)   # normalised units: std(u)=1
            u = torch.tensor(ub[i][None, :] + eps, dtype=torch.float32, device=device)
            ys = torch.cat([net((u[j:j + bs], xt))[:, :N] for j in range(0, M, bs)])
            out.append(ys.std(dim=0, unbiased=True).cpu().numpy())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return np.stack(out), (time.perf_counter() - t0) / len(ub) * 1e3


def conformal_from_shape(y, shape, Yt, alpha=0.10, calib_frac=0.5):
    n = len(y); nc = int(n * calib_frac)
    sc = (np.abs(Yt - y)[:nc] / shape[:nc]).ravel()
    k = int(np.ceil((sc.size + 1) * (1 - alpha))) / sc.size
    q = float(np.quantile(sc, min(k, 1.0)))
    ye, Yte, she = y[nc:], Yt[nc:], shape[nc:]
    return ye, ye - q * she, ye + q * she, Yte, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--det", required=True, help="stem of the deterministic model")
    ap.add_argument("--bands", nargs="*", default=[], help="name=stem of trained band models to add (own-conformal rescale)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--calib_test", default=None, help="h5 used only for calibration (normalisation constant + q); --test fully evaluated (OOD protocol)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mc_M", type=int, default=64, help="forward passes for the MC-perturbation post-hoc score (0 = skip)")
    a = ap.parse_args()

    ns = argparse.Namespace(data_path=a.test, num_branch_sensors=0)
    branch, trunk, Yt = read_data(ns)
    Jref = None
    with h5py.File(a.test) as f:
        if "jacobian_reference" in f:
            Jref = f["jacobian_reference"][:]
    branch, trunk, Yt = branch.astype(np.float32), trunk.astype(np.float32), Yt.astype(np.float32)

    net, args, scalers, nparams = load_model(a.det, a.device, a.iter)
    y, _, _, lat = predict_physical(net, args, scalers, branch, trunk, a.device)
    import time; t0 = time.perf_counter()
    s = jacobian_shape(net, scalers, branch, trunk, a.device)
    lat_jac = (time.perf_counter() - t0) / len(branch) * 1e3
    s_phys = s * scalers["Y_std"]                                       # bring to physical units
    smc_phys, lat_mc = (None, 0.0)
    if a.mc_M > 0:
        smc, lat_mc = mc_perturb_shape(net, scalers, branch, trunk, a.device, M=a.mc_M)
        smc_phys = smc * scalers["Y_std"]
    if a.calib_test:
        # OOD protocol: normalisation constant and q come from the calibration file; --test is fully evaluated.
        nsc = argparse.Namespace(data_path=a.calib_test, num_branch_sensors=0)
        cb, ct, cY = read_data(nsc); cb, ct, cY = cb.astype(np.float32), ct.astype(np.float32), cY.astype(np.float32)
        cy, _, _, _ = predict_physical(net, args, scalers, cb, ct, a.device)
        cs = jacobian_shape(net, scalers, cb, ct, a.device) * scalers["Y_std"]
        norm_c = cs.mean()
        if smc_phys is not None:
            csmc = mc_perturb_shape(net, scalers, cb, ct, a.device, M=a.mc_M)[0] * scalers["Y_std"]
            smc_all, norm_mc = np.concatenate([csmc, smc_phys]), csmc.mean()
        y_all, s_all, Y_all = np.concatenate([cy, y]), np.concatenate([cs, s_phys]), np.concatenate([cY, Yt])
        nc0 = len(cy)
        calib_frac = nc0 / len(y_all)
        Jref_all = np.concatenate([np.zeros_like(cY), Jref]) if Jref is not None else None
    else:
        y_all, s_all, Y_all, Jref_all = y, s_phys, Yt, Jref
        nc0 = len(y) // 2; calib_frac = 0.5; norm_c = s_phys[:nc0].mean()
        if smc_phys is not None:
            smc_all, norm_mc = smc_phys, smc_phys[:nc0].mean()
    # GLOBAL normalisation (mean over the calibration part), as in the trained model's batch-mean
    # normalisation: keeps the relative magnitude across samples, so harder instances get wider bands.
    shape = s_all / norm_c
    shape_log = np.log1p(shape)
    shape_persample = s_all / s_all.mean(axis=1, keepdims=True)        # ablation: shape only, no instance scale
    y, Yt, Jref = y_all, Y_all, Jref_all

    results = {}
    nc = nc0
    Je = Jref[nc:] if Jref is not None else None
    rows = [("const", np.ones_like(y)), ("posthoc", shape), ("posthoc_log1p", shape_log), ("posthoc_persample", shape_persample)]
    if smc_phys is not None:
        rows.append((f"mc_perturb_M{a.mc_M}", smc_all / norm_mc))
    for name, sh in rows:
        ye, lo, up, Yte, q = conformal_from_shape(y, sh, Yt, calib_frac=calib_frac)
        m = metrics(ye, lo, up, Yte, Je); m["conf_q"] = q; m["n_params"] = int(nparams)
        m["latency_ms_per_sample"] = float(lat + (lat_mc if name.startswith("mc_perturb") else (lat_jac if name != "const" else 0.0)))
        results[name] = m
        print(f"[{name}] " + ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)))
    # rho between posthoc shape and reference sensitivity / error, on the eval half
    if Jref is not None:
        from scipy.stats import spearmanr
        idx = np.random.default_rng(0).choice(shape[nc:].size, 200_000, replace=False)
        results["posthoc"]["rho_shape_vs_Jref"] = float(spearmanr(shape[nc:].ravel()[idx], np.abs(Jref[nc:]).ravel()[idx]).statistic)
    del net; torch.cuda.empty_cache()

    for pair in a.bands:
        name, stem = pair.split("=", 1)
        netb, argsb, scb, npb = load_model(stem, a.device, a.iter)
        yb, lob, upb, latb = predict_physical(netb, argsb, scb, branch, trunk, a.device)
        ye, lo, up, Yte, q = conformal_rescale(yb, lob, upb, Yt, calib_frac=calib_frac) if not a.calib_test else (None,)*5
        if a.calib_test:
            cyb, clob, cupb, _ = predict_physical(netb, argsb, scb, cb, ct, a.device)
            q = conformal_q(cyb, clob, cupb, cY); lo, up = apply_q(yb, lob, upb, q); ye, Yte = yb, Yt[nc:]
        m = metrics(ye, lo, up, Yte, Je); m["conf_q"] = q; m["n_params"] = int(npb); m["latency_ms_per_sample"] = float(latb)
        m["mse_full"] = float(np.mean((Yt[len(Yt) - len(yb):] - yb) ** 2))
        results[name + "_conf"] = m
        print(f"[{name}_conf] " + ", ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, float)))
        del netb; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out + ".json", "w"), indent=2)
    cols = ["n_params", "mse", "picp", "mpiw", "nll", "rho_err", "rho_sens", "conf_q", "latency_ms_per_sample"]
    lines = [f"All rows: split-conformal, calibrated on first half of `{os.path.basename(a.test)}`, evaluated on second half (alpha=0.10).", "",
             "| model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join("" if m.get(c) is None else (f"{m[c]:d}" if isinstance(m[c], int) else f"{m[c]:.4g}") for c in cols) + " |")
    lines += ["", "Decile PICP (per-sample MSE deciles):", "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in m["decile_picp"]) + " |")
    lines += ["", "Decile MPIW:", "| model | " + " | ".join(f"D{i}" for i in range(1, 11)) + " |", "|" + "---|" * 11]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.3g}" for v in m["decile_mpiw"]) + " |")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()