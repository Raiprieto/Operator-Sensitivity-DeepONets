"""
Sensibilidad ||d u_i / d K||_2 del forward de una corrida CUALQUIERA, entrenada
con bandas o no, y su correlacion de rango con el error.

    python protocol/jacobian_sigma.py <benchmark> [--seed N] [--lam L] [--smoke]

Para que sirve. La cabeza jacobiana entrenada mezcla dos cosas: la geometria
cruda del operador (que existe en cualquier modelo) y el moldeado que el pinball
le hace al jacobiano de la branch durante el entrenamiento. Este script mide la
primera sola, sobre el modelo determinista congelado. Si sigma ya ordena el
error sin haber entrenado nada, entonces cualquier operador preentrenado puede
dar bandas condicionadas con una capa conformal encima.

Como se calcula sin pagar n_out pases reversos. En un DeepONet cartesiano la
salida es bilineal, y_i = <T_i, B(u)> + b, de modo que

    d y_i / d u = J_B^T T_i        con J_B = dB/du  [K, n_in]
    || d y_i / d u ||^2 = T_i^T (J_B J_B^T) T_i

o sea una Gram latente de K x K (K = 100). Es la misma identidad que ya usa
jacobian_deeponet_softplus._compute_uncertainty; aqui se aplica a pesos que
nunca vieron la pinball.

Unidades. Este es el punto delicado: los scalers son POR COORDENADA
(common.fit_scalers). Con Kn = (K - m)/X_std y un = (u - m)/Y_std,

    d u_i / d K_j = Y_std_i * (d un_i / d Kn_j) / X_std_j

El Y_std_i sale de la norma (es constante en j) y multiplica sigma nodo a nodo;
el X_std_j NO sale, va dentro de la suma sobre j, asi que entra como peso de la
Gram. Por eso se reportan variantes separadas:

    phys        Y_std * || (J_B / X_std)^T T ||     sensibilidad fisica
    norm        || J_B^T T ||                       lo que ve la red, sin escalas
    band_proxy  Y_std * log1p(norm / ref)           la forma exacta de la banda
                                                    entrenada (ver mas abajo)
    ystd        Y_std repetido                      CONTROL

El control importa: en Darcy Y_std tiene forma de campana y por si solo ya
correlaciona con el error, asi que una rho alta de 'phys' no significa nada si
'ystd' saca lo mismo.

Autochequeo. Sobre una corrida con banda de verdad, 'band_proxy' es exactamente
la forma que el forward entrenado le da al ancho, asi que su Spearman contra el
ancho observado tiene que dar ~1.000. Si no, el jacobiano de aqui esta mal.

Nota sobre el log1p: Spearman es invariante a transformaciones monotonas, de
modo que 'phys' y 'norm' no cambiarian si se les aplicara log1p. Deja de ser
inocuo solo al multiplicar por Y_std (la compresion cambia el orden ENTRE
nodos) y al construir la banda conformal, donde el multiplicador es uno solo.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402

JAC_BATCH = 16          # el jacobiano de la branch es [b, K, n_in]: batch chico
NODE_CHUNK = 2048       # nodos por trozo: acota [b, n_out, K] sin cambiar el resultado


def branch_jacobian_sigma(net, b, t, sc, device, batch=JAC_BATCH):
    """Devuelve (sigma_phys, sigma_norm), ambos [n_muestras, n_nodos].

    sigma_norm = || J_B^T T_i ||                   (normalizado, sin escalas)
    sigma_phys = Y_std * || (J_B/X_std)^T T_i ||   (unidades fisicas)

    La Gram se arma en float64: J_B J_B^T eleva al cuadrado y en float32 la
    varianza sale ligeramente negativa por redondeo (de ahi el relu que el
    modelo entrenado necesita). En doble no hace falta pelear con eso.
    """
    from torch.func import jacrev, vmap

    bs = (b - sc["branch_mean"]) / sc["branch_std"]
    tt = torch.tensor((t - sc["trunk_mean"]) / sc["trunk_std"], dtype=torch.float32, device=device)
    xstd = torch.tensor(sc["branch_std"], dtype=torch.float64, device=device)      # [n_in]
    ystd = torch.tensor(sc["Y_std"], dtype=torch.float64, device=device)           # [n_out]

    with torch.no_grad():
        T = net.trunk(tt)
        if hasattr(net, "activation_trunk"):
            T = net.activation_trunk(T)
        T = T.double()                                                             # [n_out, K]

    # El centro de la prediccion es <T_i, B[:K]>. quantile_ux emite 3K coeficientes
    # (centro, inferior, superior) con la misma base del trunk, asi que la
    # sensibilidad del centro solo involucra el primer bloque.
    K = T.shape[1]

    def _branch_one(u_s):
        return net.branch(u_s.unsqueeze(0)).squeeze(0)[:K]

    sp, sn = [], []
    for i in range(0, len(bs), batch):
        u = torch.tensor(bs[i:i + batch], dtype=torch.float32, device=device)
        # detach: jacrev deja el jacobiano enganchado al grafo de los pesos y aqui
        # no se deriva nada mas, solo se mide.
        J = vmap(jacrev(_branch_one))(u).double().detach()                         # [b, K, n_in]
        for J_w, dest, scale in ((J, sn, None), (J / xstd, sp, ystd)):
            G = torch.bmm(J_w, J_w.transpose(1, 2))                                # [b, K, K]
            # El paso intermedio es [b, n_out, K]; con los 12800 nodos de Burgers
            # eso son cientos de MB, asi que se trocea por nodos.
            var = torch.cat([
                torch.einsum("bnm,nm->bn", torch.einsum("nk,bkm->bnm", Tc, G), Tc)
                for Tc in T.split(NODE_CHUNK, dim=0)], dim=1).clamp(min=0.0)
            s = var.sqrt()
            dest.append((s * scale if scale is not None else s).cpu().numpy())
    return np.concatenate(sp, 0), np.concatenate(sn, 0)


def rho(a, b_):
    return float(spearmanr(np.asarray(a).ravel(), np.asarray(b_).ravel())[0])


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
        data_md5 = C.md5_file(data_path)
        out_name, n_ev = f"ood_{a.ood}_sigma.json", cfg["ood"][a.ood]["n"]
    else:
        C.require_data_matches_manifest(cfg, man, ("test",))
        data_path, data_md5 = C.split_path(cfg, "test"), man["files"]["test"]["md5"]
        out_name, n_ev = "jacobian_sigma.json", cfg["data"]["splits"]["test"]["n"]
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    bte_raw, t, yte = C.load_split(data_path, cfg["data"]["append_params"], n_ev)
    with h5py.File(data_path, "r") as f:
        jac = f["jacobian_reference"][:n_ev] if "jacobian_reference" in f else None
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
        err = np.abs(yte - c)

        s_phys, s_norm = branch_jacobian_sigma(net, bte, t, sc, device)
        # Invariancia al tamano de batch del jacobiano (las primeras muestras)
        k = min(8, len(bte))
        s_alt, _ = branch_jacobian_sigma(net, bte[:k], t, sc, device, batch=3)
        drift = float(np.abs(s_phys[:k] - s_alt).max() / max(np.abs(s_phys[:k]).max(), 1e-30))
        if drift > 1e-6:
            raise C.ProtocolError(f"{d}: sigma cambia con el tamano de batch ({drift:.2e})")

        ystd = np.broadcast_to(sc["Y_std"], s_phys.shape)
        ref = float(s_norm.mean())
        band_proxy = ystd * np.log1p(s_norm / (ref + 1e-30))
        # 'jac' es la forma cruda que usa jacobian_conformal: Y_std * sigma_norm,
        # sin la compresion logaritmica de band_proxy.
        variantes = {"phys": s_phys, "norm": s_norm, "jac": ystd * s_norm,
                     "band_proxy": band_proxy, "ystd": ystd}

        res = {"model": kind, "lambda": lam, "seed": seed, "mse": float((err ** 2).mean()),
               "sigma_ref_norm": ref, "sigma_batch_rel_drift": drift,
               "rho_err": {k_: rho(v, err) for k_, v in variantes.items()}}
        if jac is not None:
            res["rho_sens"] = {k_: rho(v, jac) for k_, v in variantes.items()}
        # Autochequeo: en un modelo con banda, band_proxy es su misma forma
        w = up - lo
        if w.max() > 1e-8 * max(np.abs(c).max(), 1e-30):
            res["selftest_rho_width_vs_band_proxy"] = rho(band_proxy, w)
            res["rho_err_width"] = rho(w, err)
        res.update({"commit_eval": commit, "dataset": os.path.basename(data_path),
                    "data_md5": data_md5})
        C.write_json(out, res)
        rows.append(res)
        st = res.get("selftest_rho_width_vs_band_proxy")
        print(f"  {os.path.basename(d):26s} rhoErr phys={res['rho_err']['phys']:+.3f} "
              f"norm={res['rho_err']['norm']:+.3f} ystd={res['rho_err']['ystd']:+.3f}"
              + (f"  [autochequeo {st:.4f}]" if st is not None else ""), flush=True)

    print("\n| Modelo | l | Semilla | MSE | rhoErr phys | rhoErr norm | rhoErr band_proxy "
          "| rhoErr ystd (control) | rhoErr ancho | rhoSens phys | autochequeo |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        def g(dd, k_):
            return f"{r[dd][k_]:+.3f}" if dd in r else "-"
        anc = r.get("rho_err_width")
        st = r.get("selftest_rho_width_vs_band_proxy")
        print(f"| {r['model']} | {r['lambda']:g} | {r['seed']} | {r['mse']:.3e} | "
              f"{g('rho_err', 'phys')} | {g('rho_err', 'norm')} | {g('rho_err', 'band_proxy')} | "
              f"{g('rho_err', 'ystd')} | "
              f"{'-' if anc is None else format(anc, '+.3f')} | {g('rho_sens', 'phys')} | "
              f"{'-' if st is None else format(st, '.4f')} |")


if __name__ == "__main__":
    main()
