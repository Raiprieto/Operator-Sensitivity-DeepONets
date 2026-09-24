"""
Entrena UNA corrida del barrido declarado en protocol/configs/<benchmark>.json.

    python protocol/train.py <benchmark> --index I [--smoke]
    python protocol/train.py <benchmark> --model jacobian --lam 4 --seed 0 [--smoke]

Garantias (si alguna no se cumple, aborta con codigo 3):
  - Corre dentro de Slurm, con GPU, desde codigo commiteado.
  - Los datos de train y val coinciden byte a byte con el manifiesto verificado.
  - Nunca lee el test.
  - La normalizacion se ajusta solo con el train.
  - La seleccion del checkpoint se hace SOLO con validacion (interval score).
  - La carpeta de la corrida no existe de antes: nada se sobrescribe.
  - Un NaN o infinito en validacion detiene la corrida y la marca como fallida.

Inferencia: el modelo Jacobian normaliza la forma de la incertidumbre por la
media del batch. En validacion y en la evaluacion final se usa en cambio una
referencia fija (media sobre el train), para que el intervalo de cada muestra
no dependa de las demas muestras del batch.
"""

import argparse
import csv
import os
import platform
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import deepxde as dde  # noqa: E402
import torch  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--index", type=int)
    ap.add_argument("--model", choices=list(C.MODELS))
    ap.add_argument("--lam", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main():
    a = parse()
    C.require_slurm()
    commit = C.require_clean_code()
    cfg = C.load_config(a.benchmark, a.smoke)

    runs = C.plan(cfg)
    if a.index is not None:
        if not 0 <= a.index < len(runs):
            raise C.ProtocolError(f"Indice {a.index} fuera del barrido (0..{len(runs) - 1})")
        kind, lam, seed = runs[a.index]
    else:
        kind, lam, seed = a.model, float(a.lam), int(a.seed)
        if (kind, lam, seed) not in runs:
            raise C.ProtocolError(f"({kind}, {lam}, {seed}) no esta declarado en el barrido: {runs}")

    man = C.load_manifest(cfg)
    C.require_data_matches_manifest(cfg, man, ("train", "val"))

    run_dir = os.path.join(C.runs_dir(cfg), C.run_name(kind, lam, seed))
    C.require_absent(run_dir)
    os.makedirs(run_dir)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = {"status": "running", "benchmark": cfg["benchmark"], "model": kind, "lambda": lam,
            "seed": seed, "smoke": cfg["_smoke"], "commit": commit, "config": cfg,
            "data_md5": {s: man["files"][s]["md5"] for s in C.SPLITS},
            "sigma": cfg["model"]["sigma"].get(kind), "architecture": cfg["model"],
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "host": platform.node(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    C.write_json(meta_path, meta)

    try:
        train(cfg, kind, lam, seed, run_dir, meta)
    except BaseException as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        C.write_json(meta_path, meta)
        raise
    meta["status"] = "completed"
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    C.write_json(meta_path, meta)
    print(f"\nOK: corrida completa en {run_dir}")


def train(cfg, kind, lam, seed, run_dir, meta):
    from deterministic_deeponet import mse_loss_triple
    from jacobian_deeponet_softplus import pinball_loss_softplus

    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"
    dde.config.set_default_float("float32")
    dde.config.set_random_seed(seed)
    meta["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                        "deepxde": dde.__version__, "gpu": torch.cuda.get_device_name(0)}

    tc, sp, ap_ = cfg["train"], cfg["data"]["splits"], cfg["data"]["append_params"]
    btr, t, ytr = C.load_split(C.split_path(cfg, "train"), ap_, sp["train"]["n"])
    bva, tva, yva = C.load_split(C.split_path(cfg, "val"), ap_, sp["val"]["n"])
    btr = C.transform_branch(btr, cfg, cfg["model"]["branch"][0])
    bva = C.transform_branch(bva, cfg, cfg["model"]["branch"][0])
    if not np.array_equal(t, tva):
        raise C.ProtocolError("La grilla de val no coincide con la del train")

    # Normalizacion ajustada SOLO con el train
    sc = C.fit_scalers(btr, t, ytr)
    np.savez(os.path.join(run_dir, "scalers.npz"), **sc)
    nb = lambda b: (b - sc["branch_mean"]) / sc["branch_std"]           # noqa: E731
    ny = lambda y: (y - sc["Y_mean"]) / sc["Y_std"]                     # noqa: E731
    tn = (t - sc["trunk_mean"]) / sc["trunk_std"]

    k = min(len(bva), tc["batch_size"])
    data = dde.data.TripleCartesianProd(X_train=(nb(btr), tn), y_train=ny(ytr),
                                        X_test=(nb(bva[:k]), tn), y_test=ny(yva[:k]))
    net = C.build_net(cfg["model"], kind, meta["sigma"])
    model = dde.Model(data, net)
    # El determinista no tiene bandas que entrenar: MSE puro, lambda no interviene
    loss = mse_loss_triple if kind == "mse" else partial(pinball_loss_softplus, pinball_lambda=lam)
    model.compile("adam", lr=tc["learning_rate"], loss=loss,
                  decay=("step", tc["decay_steps"], tc["decay_rate"]))

    # Subconjunto fijo del train para la referencia de incertidumbre en validacion
    rng = np.random.default_rng(seed)
    ref_idx = np.sort(rng.choice(len(btr), size=min(tc["unc_ref_subset"], len(btr)), replace=False))
    sel = ValSelector(net, kind, btr[ref_idx], bva, yva, t, sc, device, tc, run_dir)

    t0 = time.time()
    losshistory, _ = model.train(iterations=tc["iterations"], batch_size=tc["batch_size"],
                                 display_every=tc["display_every"], callbacks=[sel])
    meta["train_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(run_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss_train", "loss_val_subset"])
        for s, lt, lv in zip(losshistory.steps, losshistory.loss_train, losshistory.loss_test):
            w.writerow([s, float(np.sum(lt)), float(np.sum(lv))])

    if sel.best_step is None:
        raise C.ProtocolError("No se selecciono ningun checkpoint (sin evaluaciones validas).")

    # Checkpoint seleccionado + referencia de incertidumbre sobre TODO el train
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    unc_ref = None
    if kind in C.JAC_MODELS:
        net.unc_ref = None
        unc_ref = C.compute_unc_ref(net, btr, t, sc, device, tc["val_infer_batch"])
        net.unc_ref = unc_ref
    c, lo, up = C.predict(net, bva, t, sc, device, tc["val_infer_batch"])
    final_val = C.metrics(yva, c, lo, up)
    if final_val["nonfinite"]:
        raise C.ProtocolError(f"Metricas de validacion no finitas en el checkpoint final: {final_val}")

    meta.update({"best_step": sel.best_step, "best_val_subset_ref": sel.best_metrics,
                 "val": final_val, "unc_ref": unc_ref, "unc_ref_n": len(btr) if unc_ref else None,
                 "stopped_early": bool(model.stop_training), "iterations_run": int(model.train_state.step),
                 "selection_metric": tc["selection_metric"]})
    print(f"Checkpoint seleccionado: paso {sel.best_step} | val {tc['selection_metric']}="
          f"{final_val[tc['selection_metric']]:.4e} PICP={final_val['picp']:.2f}% MSE={final_val['mse']:.3e}")


class ValSelector(dde.callbacks.Callback):
    """Callback de DeepXDE: valida cada `val_every` pasos y guarda el mejor checkpoint."""

    def __init__(self, net, kind, b_ref, bva, yva, t, sc, device, tc, run_dir):
        super().__init__()
        self.net, self.kind, self.b_ref, self.bva, self.yva = net, kind, b_ref, bva, yva
        self.t, self.sc, self.device, self.tc, self.run_dir = t, sc, device, tc, run_dir
        self.metric = tc["selection_metric"]
        self.best, self.best_step, self.best_metrics, self.bad = np.inf, None, None, 0
        self.csv = os.path.join(run_dir, "val_history.csv")
        with open(self.csv, "w", newline="") as f:
            csv.writer(f).writerow(["step", "mse", "picp", "mpiw", "nll", "interval_score",
                                    "unc_ref_subset", "selected"])

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.tc["val_every"] == 0 or step == self.tc["iterations"]:
            self.validate(step)

    def validate(self, step):
        ref = None
        if self.kind in C.JAC_MODELS:
            self.net.unc_ref = None
            ref = C.compute_unc_ref(self.net, self.b_ref, self.t, self.sc, self.device,
                                    self.tc["val_infer_batch"])
            self.net.unc_ref = ref
        self.net.eval()                  # congela medias moviles (correccion centrada)
        try:
            c, lo, up = C.predict(self.net, self.bva, self.t, self.sc, self.device,
                                  self.tc["val_infer_batch"])
        finally:
            self.net.unc_ref = None      # el entrenamiento SIEMPRE usa la media del batch
            self.net.train()
        m = C.metrics(self.yva, c, lo, up)
        if m["nonfinite"]:
            raise C.ProtocolError(f"Validacion no finita en el paso {step}: {m}")
        score = m[self.metric]
        improved = step >= self.tc["min_select_step"] and score < self.best
        if improved:
            self.best, self.best_step, self.best_metrics, self.bad = score, step, m, 0
            torch.save({"model_state_dict": self.net.state_dict(), "step": step},
                       os.path.join(self.run_dir, "best.pt"))
        else:
            self.bad += 1
        with open(self.csv, "a", newline="") as f:
            csv.writer(f).writerow([step, m["mse"], m["picp"], m["mpiw"], m["nll"],
                                    m["interval_score"], ref, int(improved)])
        print(f"  [val paso {step}] {self.metric}={score:.4e} PICP={m['picp']:.2f}% "
              f"MSE={m['mse']:.3e}{'  *mejor*' if improved else ''}", flush=True)
        pat = self.tc.get("patience")
        if pat and self.bad >= pat:
            print(f"  Early stopping: {pat} validaciones sin mejora.", flush=True)
            self.model.stop_training = True


if __name__ == "__main__":
    main()
