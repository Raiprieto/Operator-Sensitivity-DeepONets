"""
input_noise_propagation.py -- the task only a Jacobian can do: propagate a KNOWN input uncertainty.

Setting (Burgers 1D): the initial condition is measured with Gaussian noise  u0_obs = u0 + eps,
eps ~ N(0, sigma^2 I).  We want the standard deviation of the PDE solution field induced by that noise.

Ground truth : Monte Carlo through the reference FVM solver (M noisy ICs, M full simulations).
Estimators   : (a) delta method with the latent Jacobian of a trained DeepONet, ONE pass:
                   Var(x) = T(x) (J_branch S J_branch^T) T(x)^T,  S = diag(sigma^2 / branch_std^2) on the u0 sensors;
               (b) Monte Carlo through the DeepONet (M forward passes).
Any branch/trunk backbone can be used (det / qux / jac); quantile heads give no such quantity.

Metrics per (model, sigma): relative L2 error of the std field vs solver truth (mean over inputs),
pointwise Spearman(std_est, std_truth), wall time per input for each estimator.

Usage:
    python results/input_noise_propagation.py --test data/burgers_test_500.h5 --iter 100000 \
        --models det=modelos/burgers_fair/det qux=modelos/burgers_fair/qux jac=modelos/burgers_fair/jac \
        --out results/fair/noise_prop_burgers_it100000
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src", "data_generation"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from fair_eval import load_model, read_data  # noqa: E402
import jax, jax.numpy as jnp  # noqa: E402
from burgers_pdebench import solve_burgers_batch  # noqa: E402


def solver_truth(u0, nu, sigma, M, nx, nt, L, T, rng):
    eps = rng.normal(0.0, sigma, size=(M, nx)).astype(np.float32)
    u0b = jnp.asarray(u0[None, :] + eps)
    nub = jnp.full((M,), float(nu), dtype=jnp.float32)
    t0 = time.perf_counter()
    sol = solve_burgers_batch(u0b, nx, L / nx, T / nt, nt, nub)          # [M, nt, nx]
    sol = np.asarray(sol).transpose(0, 2, 1).reshape(M, -1)                # x-major like the trunk
    dt = time.perf_counter() - t0
    return sol.std(axis=0, ddof=1), dt


def backbone_forward(net, u, x):
    out = net((u, x))
    N = x.shape[0]
    return out[:, :N]


def delta_std(net, scalers, branch_row, trunk_scaled_t, sigma, n_u0, device):
    """One-pass delta-method std field (physical units) for one input."""
    from torch.func import vmap, jacrev
    ub = torch.tensor(((branch_row - scalers["branch_mean"]) / scalers["branch_std"])[None, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        Tm = net.activation_trunk(net.trunk(trunk_scaled_t))                         # [N, K]
    def branch_single(u_s):
        return net.branch(u_s.unsqueeze(0)).squeeze(0)
    t0 = time.perf_counter()
    J = vmap(jacrev(branch_single))(ub)[0][:Tm.shape[1]]                              # [K, N_in] (split-branch: centre head only)
    s = torch.zeros(J.shape[1], device=device)
    s[:n_u0] = torch.tensor(sigma / scalers["branch_std"][:n_u0], dtype=torch.float32, device=device)
    Js = J * s[None, :]
    G = Js @ Js.T                                                                     # [K, K]
    var = torch.einsum("nk,km,nm->n", Tm, G, Tm).clamp_min(0)
    std = torch.sqrt(var).detach().cpu().numpy() * scalers["Y_std"]
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return std, time.perf_counter() - t0


def mc_std(net, scalers, branch_row, trunk_scaled_t, sigma, n_u0, M, device, rng, bs=64):
    eps = rng.normal(0.0, sigma, size=(M, n_u0)).astype(np.float32)
    rows = np.repeat(branch_row[None, :], M, axis=0); rows[:, :n_u0] += eps
    ub = torch.tensor((rows - scalers["branch_mean"]) / scalers["branch_std"], dtype=torch.float32, device=device)
    outs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, M, bs):
            outs.append(backbone_forward(net, ub[i:i + bs], trunk_scaled_t).cpu().numpy())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    y = np.concatenate(outs) * scalers["Y_std"] + scalers["Y_mean"]
    return y.std(axis=0, ddof=1), dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--iter", type=int, default=None)
    ap.add_argument("--n_inputs", type=int, default=16)
    ap.add_argument("--n_mc", type=int, default=256)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    import h5py
    with h5py.File(a.test) as f:
        nx, nt, L, T = int(f.attrs["nx"]), int(f.attrs["nt"]), float(f.attrs["L"]), float(f.attrs["T"])
    ns = argparse.Namespace(data_path=a.test, num_branch_sensors=0)
    branch, trunk, _ = read_data(ns)
    branch, trunk = branch.astype(np.float32), trunk.astype(np.float32)
    n_u0 = nx
    idx = np.arange(a.n_inputs)
    rng = np.random.default_rng(0)

    # ground truth (solver MC), cached per sigma
    truth = {}
    for sg in a.sigmas:
        stds, times = [], []
        for i in idx:
            s, dt = solver_truth(branch[i, :n_u0], branch[i, n_u0], sg, a.n_mc, nx, nt, L, T, rng)
            stds.append(s); times.append(dt)
        truth[sg] = (np.stack(stds), float(np.mean(times[1:])))          # skip JIT warm-up time
        print(f"[solver MC] sigma={sg}: {truth[sg][1]*1e3:.0f} ms/input for M={a.n_mc}")

    results = {"solver_mc_ms_per_input": {str(sg): truth[sg][1] * 1e3 for sg in a.sigmas}, "n_mc": a.n_mc, "n_inputs": a.n_inputs}
    for pair in a.models:
        name, stem = pair.split("=", 1)
        net, args, scalers, _ = load_model(stem, a.device, a.iter)
        xt = torch.tensor((trunk - scalers["trunk_mean"]) / scalers["trunk_std"], dtype=torch.float32, device=a.device)
        for sg in a.sigmas:
            d_std, d_t, m_std, m_t = [], [], [], []
            for i in idx:
                s, dt = delta_std(net, scalers, branch[i], xt, sg, n_u0, a.device); d_std.append(s); d_t.append(dt)
                s, dt = mc_std(net, scalers, branch[i], xt, sg, n_u0, a.n_mc, a.device, rng); m_std.append(s); m_t.append(dt)
            d_std, m_std = np.stack(d_std), np.stack(m_std)
            tr = truth[sg][0]
            rel = lambda e: float(np.mean(np.linalg.norm(e - tr, axis=1) / np.linalg.norm(tr, axis=1)))
            rho = lambda e: float(spearmanr(e.ravel(), tr.ravel()).statistic)
            results[f"{name}|sigma={sg}"] = {
                "delta_relL2": rel(d_std), "delta_rho": rho(d_std), "delta_ms": float(np.mean(d_t[1:])) * 1e3,
                "mc_relL2": rel(m_std), "mc_rho": rho(m_std), "mc_ms": float(np.mean(m_t[1:])) * 1e3,
                "delta_vs_mc_relL2": float(np.mean(np.linalg.norm(d_std - m_std, axis=1) / np.linalg.norm(m_std, axis=1))),
            }
            r = results[f"{name}|sigma={sg}"]
            print(f"[{name} sigma={sg}] delta: relL2={r['delta_relL2']:.3f} rho={r['delta_rho']:.3f} {r['delta_ms']:.1f} ms | "
                  f"MC(M={a.n_mc}): relL2={r['mc_relL2']:.3f} rho={r['mc_rho']:.3f} {r['mc_ms']:.0f} ms | delta-vs-MC {r['delta_vs_mc_relL2']:.3f}")
        del net; torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out + ".json", "w"), indent=2)
    lines = [f"Input-noise propagation, Burgers IC noise eps~N(0,sigma^2 I); truth = solver MC (M={a.n_mc}, {a.n_inputs} inputs).",
             "solver MC cost per input: " + ", ".join(f"sigma={sg}: {truth[sg][1]*1e3:.0f} ms" for sg in a.sigmas), "",
             "| backbone | sigma | delta relL2 | delta rho | delta ms | DeepONet-MC relL2 | MC rho | MC ms | delta vs MC relL2 |", "|" + "---|" * 9]
    for k, r in results.items():
        if "|sigma=" in k:
            name, sg = k.split("|sigma=")
            lines.append(f"| {name} | {sg} | {r['delta_relL2']:.3f} | {r['delta_rho']:.3f} | {r['delta_ms']:.1f} | {r['mc_relL2']:.3f} | {r['mc_rho']:.3f} | {r['mc_ms']:.0f} | {r['delta_vs_mc_relL2']:.3f} |")
    open(a.out + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()