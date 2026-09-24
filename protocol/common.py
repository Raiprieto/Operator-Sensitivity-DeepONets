"""
Nucleo del protocolo de entrenamiento y evaluacion.

Principio: el protocolo NO advierte y sigue; ABORTA. Cualquier violacion
(codigo sin commitear, datos que no coinciden con el manifiesto, solapamiento
entre splits, rangos de semilla que se cruzan, NaN, sobrescribir resultados,
correr fuera de Slurm) termina el proceso con codigo de salida != 0.
"""

import copy
import hashlib
import json
import os
import subprocess
import sys

import h5py
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (REPO, os.path.join(REPO, "deepxde-extensions"),
           os.path.join(REPO, "src", "data_generation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MODELS = ("jacobian", "jacobian_quantile", "jacobian_quantile_x", "jacobian_quantile_c",
          "jacobian_quantile_cx", "vanilla", "quantile_x", "quantile_ux", "mse")
SIGMA_MODELS = ("jacobian", "jacobian_quantile", "jacobian_quantile_x", "jacobian_quantile_c",
                "jacobian_quantile_cx", "vanilla")   # los quantile puros no usan sigma
JAC_MODELS = ("jacobian", "jacobian_quantile", "jacobian_quantile_x", "jacobian_quantile_c",
              "jacobian_quantile_cx")   # usan ||J|| y por tanto referencia de incertidumbre
ALPHA = 0.10            # intervalo nominal del 90 %
Z90 = 1.6448536269514722
SPLITS = ("train", "val", "test")
# Rutas cuyo codigo determina los resultados: deben estar commiteadas.
CODE_PATHS = ("src", "deepxde-extensions", "protocol",
              # MODELS.md es un indice generado: no determina ningun resultado y se
              # reescribe en el cluster, asi que no puede bloquear los jobs.
              ":(exclude)protocol/MODELS.md")


class ProtocolError(SystemExit):
    """Violacion del protocolo: termina el proceso con codigo 3."""

    def __init__(self, msg):
        print(f"\n[PROTOCOLO VIOLADO] {msg}", file=sys.stderr, flush=True)
        super().__init__(3)


# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

SMOKE = {
    "n": {"train": 256, "val": 64, "test": 64},
    "train": {"iterations": 400, "val_every": 200, "display_every": 100,
              "unc_ref_subset": 64},
}


def load_config(benchmark, smoke=False):
    path = os.path.join(REPO, "protocol", "configs", f"{benchmark}.json")
    if not os.path.exists(path):
        raise ProtocolError(f"No existe la configuracion {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_smoke"] = bool(smoke)
    base = None
    if "data_from" in cfg:
        # Variante (experimento): usa los datos ya verificados de otra configuracion.
        if "data" in cfg:
            raise ProtocolError(f"'{benchmark}' declara data_from y no puede redefinir 'data'")
        base = load_config(cfg["data_from"], smoke)
        cfg["data"] = copy.deepcopy(base["data"])
        if "ood" not in cfg and "ood" in base:      # las variantes heredan los conjuntos OOD
            cfg["ood"] = copy.deepcopy(base["ood"])
    if smoke:
        cfg = copy.deepcopy(cfg)
        for s in SPLITS if base is None else ():
            want = SMOKE["n"][s]
            if s == "train":   # al menos un batch completo, o DeepXDE no puede muestrear
                want = max(want, cfg["train"]["batch_size"])
            cfg["data"]["splits"][s]["n"] = min(cfg["data"]["splits"][s]["n"], want)
        cfg["train"].update(SMOKE["train"])
        cfg["sweep"] = {"models": cfg["sweep"]["models"],
                        "lambdas": cfg["sweep"]["lambdas"][:1], "seeds": [0]}
    validate_config(cfg)
    # Firma de la configuracion efectiva: cualquier cambio (del archivo o de los
    # overrides de smoke) invalida el manifiesto y obliga a verificar de nuevo.
    eff = {k: v for k, v in cfg.items() if not k.startswith("_")}
    cfg["_config_md5"] = _md5_json(eff)
    # El manifiesto certifica los DATOS: solo lo que los define. Cambiar el
    # barrido o el optimizador no invalida la verificacion; cambiar un split, una
    # semilla o la dimension de entrada si.
    if base is not None:
        cfg["_data_benchmark"] = base["_data_benchmark"]
        cfg["_data_config_md5"] = base["_data_config_md5"]
    else:
        cfg["_data_benchmark"] = cfg["benchmark"]
        cfg["_data_config_md5"] = _md5_json({"benchmark": cfg["benchmark"], "smoke": cfg["_smoke"],
                                             "data": cfg["data"], "branch_dim": cfg["model"]["branch"][0]})
    return cfg


def _md5_json(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def seed_span(split_cfg):
    """Rango [lo, hi) de enteros de semilla PRNG que consume un split generado."""
    g, s, n = split_cfg["generator"], split_cfg["seed"], split_cfg["n"]
    if g == "darcy":
        return s, s + n                       # PRNGKey(collected + offset), offset fijo
    if g in ("burgers", "ns2d"):
        # offset crece con cada lote (tambien los descartados) y los parametros
        # usan +10000 / +20000. Cota conservadora: descartes <= n.
        return s, s + 4 * n + 30000
    if g == "biharmonic_scaled":
        return s, s + 1                       # semilla de numpy: un unico flujo
    raise ProtocolError(f"Generador desconocido: {g}")


def validate_config(cfg):
    sp = cfg["data"]["splits"]
    if set(sp) != set(SPLITS):
        raise ProtocolError(f"Los splits deben ser exactamente {SPLITS}, hay {sorted(sp)}")
    gen = {k: v for k, v in sp.items() if "generator" in v}
    spans = {k: seed_span(v) for k, v in gen.items()}
    ks = sorted(spans)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if sp[a]["generator"] != sp[b]["generator"]:
                continue
            (a0, a1), (b0, b1) = spans[a], spans[b]
            if a0 < b1 and b0 < a1:
                raise ProtocolError(
                    f"Rangos de semilla solapados entre '{a}' {spans[a]} y '{b}' {spans[b]}")
    for m in cfg["sweep"]["models"]:
        if m not in MODELS:
            raise ProtocolError(f"Modelo desconocido: {m}. Soportados: {MODELS}")
        if m in SIGMA_MODELS and m not in cfg["model"]["sigma"]:
            raise ProtocolError(f"Falta sigma para el modelo '{m}'")
    if sp["train"]["n"] < cfg["train"]["batch_size"]:
        raise ProtocolError(f"n_train={sp['train']['n']} < batch_size={cfg['train']['batch_size']}")
    for s in ("val", "test"):
        if sp[s]["n"] < 1:
            raise ProtocolError(f"El split '{s}' no tiene muestras")
    if not cfg["sweep"]["seeds"] or not cfg["sweep"]["lambdas"]:
        raise ProtocolError("El barrido debe declarar semillas y lambdas")
    # Los conjuntos OOD tambien deben usar rangos de semilla propios
    for name, o in cfg.get("ood", {}).items():
        lo, hi = seed_span(o)
        for k, v in sp.items():
            if "generator" not in v or v["generator"] != o["generator"]:
                continue
            a0, a1 = seed_span(v)
            if lo < a1 and a0 < hi:
                raise ProtocolError(f"El OOD '{name}' {(lo, hi)} se cruza con el split '{k}' {(a0, a1)}")


def data_dir(cfg):
    root = "protocol_smoke" if cfg["_smoke"] else "protocol"
    return os.path.join(REPO, "data", root, cfg.get("_data_benchmark", cfg["benchmark"]))


def runs_dir(cfg):
    root = "protocol_smoke" if cfg["_smoke"] else "protocol"
    return os.path.join(REPO, "runs", root, cfg["benchmark"])


def split_path(cfg, split):
    s = cfg["data"]["splits"][split]
    if "file" in s:
        return os.path.join(REPO, s["file"])
    return os.path.join(data_dir(cfg), f"{split}.h5")


def ood_path(cfg, name):
    if name not in cfg.get("ood", {}):
        raise ProtocolError(f"'{name}' no esta declarado en la seccion ood de la config")
    return os.path.join(data_dir(cfg), f"ood_{name}.h5")


def manifest_path(cfg):
    return os.path.join(data_dir(cfg), "MANIFEST.json")


def run_name(model, lam, seed):
    return f"{model}_lam{lam:g}_seed{seed}"


def plan(cfg):
    """Lista ordenada y determinista de corridas declaradas en el barrido."""
    return [(m, float(l), int(s)) for m in cfg["sweep"]["models"]
            for l in cfg["sweep"]["lambdas"] for s in cfg["sweep"]["seeds"]]


# --------------------------------------------------------------------------
# Guardias
# --------------------------------------------------------------------------

def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def require_slurm():
    if not os.environ.get("SLURM_JOB_ID"):
        raise ProtocolError("Solo se ejecuta dentro de un job de Slurm (nunca en el nodo de login).")


def require_clean_code():
    """Devuelve el commit actual; aborta si el codigo relevante no esta commiteado."""
    def git(*a):
        return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = git("status", "--porcelain", "--", *CODE_PATHS).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ProtocolError(f"No se pudo consultar git: {e}")
    if dirty:
        raise ProtocolError("Hay cambios sin commitear en el codigo que determina los resultados:\n"
                            + dirty + "\nCommitea (y sincroniza por git) antes de correr.")
    return commit


def require_absent(path):
    if os.path.exists(path):
        raise ProtocolError(f"Ya existe {path}. El protocolo nunca sobrescribe resultados.")


def load_manifest(cfg):
    p = manifest_path(cfg)
    if not os.path.exists(p):
        raise ProtocolError(f"Falta {p}. Corre protocol/verify.py antes de entrenar.")
    with open(p, encoding="utf-8") as f:
        man = json.load(f)
    if "data_config_md5" in man:
        ok = man["data_config_md5"] == cfg["_data_config_md5"]
    else:   # manifiestos anteriores firmaban la config completa
        ok = man.get("config_md5") == cfg["_config_md5"]
    if not ok:
        raise ProtocolError("La definicion de los datos cambio despues de verificarlos. Regenera y verifica.")
    if not man.get("leakage_ok"):
        raise ProtocolError("El manifiesto no certifica ausencia de leakage.")
    return man


def require_data_matches_manifest(cfg, man, splits=SPLITS):
    for s in splits:
        p = split_path(cfg, s)
        if not os.path.exists(p):
            raise ProtocolError(f"Falta el archivo del split '{s}': {p}")
        got = md5_file(p)
        if got != man["files"][s]["md5"]:
            raise ProtocolError(f"El split '{s}' no coincide con el manifiesto "
                                f"(md5 {got} != {man['files'][s]['md5']}).")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------

def load_split(path, append_params, n=None):
    """Devuelve (branch, trunk, targets) en float32. Acepta formato plano y fair-sciml."""
    with h5py.File(path, "r") as f:
        if "branch_inputs" in f:
            total = f["branch_inputs"].shape[0]
            k = total if n is None else n
            if k > total:
                raise ProtocolError(f"{path} tiene {total} muestras y se piden {k}")
            b = f["branch_inputs"][:k].reshape(k, -1)
            if append_params and "params_coefficient" in f:
                p = f["params_coefficient"][:k].reshape(k, -1)
                b = np.concatenate([b, p.astype(b.dtype)], axis=1)
            t = f["trunk_inputs"][:]
            y = f["targets"][:k]
        else:
            b, y, t = [], [], None
            for s in f:
                for sim in f[s]:
                    g = f[s][sim]
                    b.append(g["field_input_f"][:])
                    y.append(g["values"][:].squeeze())
                    if t is None:
                        t = g["coordinates"][:, :2]
            b, y = np.array(b), np.array(y)
            if n is not None:
                if n > len(b):
                    raise ProtocolError(f"{path} tiene {len(b)} muestras y se piden {n}")
                b, y = b[:n], y[:n]
    return b.astype(np.float32), np.asarray(t, np.float32), y.astype(np.float32)


def transform_branch(b, cfg, expected_dim):
    """Aplica input_transform de la config y verifica la dimension que espera el modelo."""
    tr = cfg.get("input_transform")
    if tr:
        if tr.get("type") != "grid_stride":
            raise ProtocolError(f"input_transform desconocido: {tr}")
        h, w = tr["grid"]
        st = tr["stride"]
        field = b[:, :h * w].reshape(len(b), h, w)[:, ::st, ::st].reshape(len(b), -1)
        b = np.concatenate([field, b[:, h * w:]], axis=1)   # los parametros escalares se conservan
    if b.shape[1] != expected_dim:
        raise ProtocolError(f"La entrada del branch tiene {b.shape[1]} dimensiones y el modelo espera {expected_dim}")
    return np.ascontiguousarray(b)


def fit_scalers(b, t, y):
    return {"branch_mean": b.mean(0), "branch_std": b.std(0) + 1e-8,
            "trunk_mean": t.mean(0), "trunk_std": t.std(0) + 1e-8,
            "Y_mean": y.mean(0), "Y_std": y.std(0) + 1e-8}


# --------------------------------------------------------------------------
# Modelo, prediccion y metricas
# --------------------------------------------------------------------------

def build_net(model_cfg, kind, sigma):
    """Modelos soportados, todos con salida cat([y, y_lower, y_upper]):

      jacobian     ancho moldeado por la norma del jacobiano del branch (sigma)
      vanilla      ancho constante, pinball clasico (sigma)
      quantile_x   ancho espacial, depende solo de x   (baseline de la rama fair-quantile)
      quantile_ux  ancho condicionado en u y en x      (idem)
      jacobian_quantile    jacobiano + correccion aprendida por muestra y punto
      jacobian_quantile_x   jacobiano + correccion aprendida solo espacial
      jacobian_quantile_c   idem "quantile" pero con la correccion centrada
      jacobian_quantile_cx  idem "quantile_x" pero con la correccion centrada
      mse          determinista, sin bandas (base de los benchmarks Monte Carlo)
    """
    common = dict(layer_sizes_branch=model_cfg["branch"], layer_sizes_trunk=model_cfg["trunk"],
                  activation=model_cfg["activation"], kernel_initializer="Glorot normal")
    if kind in ("quantile_x", "quantile_ux"):
        from quantile_deeponet import create_spatial_quantile_deeponet
        return create_spatial_quantile_deeponet(is_cartesian=True)(
            conditioning=kind.split("_")[1], **common)
    if kind.startswith("jacobian_quantile"):
        from jacobian_quantile_deeponet import create_jacobian_quantile_deeponet
        suf = kind[len("jacobian_quantile"):]          # "", "_x", "_c", "_cx"
        return create_jacobian_quantile_deeponet(is_cartesian=True)(
            jacobian_type="parameter", sigma=sigma,
            conditioning="x" if suf.endswith("x") else "ux",
            centered=suf.startswith("_c"), **common)
    if kind == "mse":
        from deterministic_deeponet import create_deterministic_deeponet
        return create_deterministic_deeponet(is_cartesian=True)(**common)
    if kind == "jacobian":
        from jacobian_deeponet_softplus import create_jacobian_deeponet_softplus
        return create_jacobian_deeponet_softplus(is_cartesian=True)(
            jacobian_type="parameter", sigma=sigma, **common)
    from vanilla_pinball_deeponet import create_vanilla_pinball_deeponet
    return create_vanilla_pinball_deeponet(is_cartesian=True)(sigma=sigma, **common)


def predict(net, b, t, sc, device, batch):
    """Prediccion en unidades fisicas: (centro, inferior, superior)."""
    import torch
    bs = (b - sc["branch_mean"]) / sc["branch_std"]
    tt = torch.tensor((t - sc["trunk_mean"]) / sc["trunk_std"], dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(bs), batch):
            tb = torch.tensor(bs[i:i + batch], dtype=torch.float32, device=device)
            out.append(net((tb, tt)).cpu().numpy())
    p = np.concatenate(out, 0)
    n = p.shape[1] // 3
    std, mean = sc["Y_std"], sc["Y_mean"]
    return tuple(p[:, k * n:(k + 1) * n] * std + mean for k in range(3))


def pred_scale(c, lo, up):
    """Escala de referencia del test de invarianza al tamano de batch.

    Normalmente el ancho del intervalo. Un modelo determinista lo tiene en cero,
    asi que ahi se usa la amplitud de la prediccion: sin esto, cualquier
    diferencia de redondeo se dividiria por ~0 y el test abortaria siempre.
    """
    w = float(np.abs(up - lo).max())
    a = float(np.abs(c).max())
    return w if w > 1e-8 * a else max(a, 1e-12)


def compute_unc_ref(net, b, t, sc, device, batch):
    """Media de la incertidumbre (pre-normalizacion) sobre las muestras dadas.

    Es exactamente la cantidad que el forward promedia sobre el batch; aqui se
    promedia sobre el conjunto de referencia (train) para fijarla en inferencia.
    """
    import torch
    bs = (b - sc["branch_mean"]) / sc["branch_std"]
    tt = torch.tensor((t - sc["trunk_mean"]) / sc["trunk_std"], dtype=torch.float32, device=device)
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(bs), batch):
            tb = torch.tensor(bs[i:i + batch], dtype=torch.float32, device=device)
            u = net._compute_uncertainty(None, tb, tt, True)
            tot += float(u.double().sum())
            cnt += u.numel()
    ref = tot / cnt
    if not np.isfinite(ref) or ref <= 0:
        raise ProtocolError(f"Referencia de incertidumbre invalida: {ref}")
    return ref


def metrics(y, c, lo, up):
    """Metricas globales y por decil de error (unidades fisicas)."""
    mse_s = ((y - c) ** 2).mean(1)
    cov_s = ((y >= lo) & (y <= up)).mean(1) * 100
    w = up - lo
    mpiw_s = w.mean(1)
    std = w / (2 * Z90)
    var = std ** 2 + 1e-8
    nll_s = (0.5 * np.log(2 * np.pi * var) + 0.5 * (y - c) ** 2 / var).mean(1)
    # Interval score (Gneiting & Raftery 2007): regla propia para intervalos.
    is_s = (w + (2 / ALPHA) * (lo - y) * (y < lo) + (2 / ALPHA) * (y - up) * (y > up)).mean(1)
    out = {"mse": float(mse_s.mean()), "picp": float(cov_s.mean()), "mpiw": float(mpiw_s.mean()),
           "nll": float(nll_s.mean()), "interval_score": float(is_s.mean()), "n_samples": int(len(y))}
    if len(y) >= 10:
        order = np.argsort(mse_s)
        dec = np.array_split(order, 10)
        out["decile_picp"] = [float(cov_s[d].mean()) for d in dec]
        out["decile_mpiw"] = [float(mpiw_s[d].mean()) for d in dec]
    out["nonfinite"] = any(isinstance(v, float) and not np.isfinite(v) for v in out.values())
    return out


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
        f.write("\n")
    os.replace(tmp, path)
