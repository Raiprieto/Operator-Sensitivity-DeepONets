"""
Evalua en TEST las corridas completadas de un benchmark.

    python protocol/evaluate.py <benchmark> [--smoke]

Garantias (si alguna no se cumple, aborta con codigo 3):
  - El test coincide byte a byte con el manifiesto verificado.
  - Cada corrida se evalua con el sigma, la arquitectura, los scalers y la
    referencia de incertidumbre guardados en SU propia metadata: no hay forma
    de evaluar con hiperparametros distintos a los del entrenamiento.
  - La prediccion es determinista por muestra: se calcula con dos tamanos de
    batch distintos y deben coincidir.
  - Todas las corridas declaradas deben estar completadas; no se evalua un
    subconjunto.
  - Un resultado ya escrito nunca se sobrescribe: las corridas ya evaluadas
    (sobre el mismo test, segun su md5) se saltan.

Ademas del modo principal (referencia fija), reporta el modo original del paper
(media del batch, batch=50) solo como referencia comparativa.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402
import torch  # noqa: E402

LEGACY_BATCH = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    C.require_slurm()
    commit = C.require_clean_code()
    cfg = C.load_config(a.benchmark, a.smoke)
    man = C.load_manifest(cfg)
    C.require_data_matches_manifest(cfg, man, ("test",))
    if not torch.cuda.is_available():
        raise C.ProtocolError("No hay GPU disponible en este job.")
    device = "cuda"

    runs = C.plan(cfg)
    dirs = [os.path.join(C.runs_dir(cfg), C.run_name(*r)) for r in runs]
    metas = []
    for d in dirs:
        mp = os.path.join(d, "metadata.json")
        if not os.path.exists(mp):
            raise C.ProtocolError(f"Falta la corrida {d}")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m["status"] != "completed":
            raise C.ProtocolError(f"La corrida {d} no esta completada (status={m['status']}). "
                                  "No se evalua un subconjunto del barrido.")
        if m["data_md5"] != {s: man["files"][s]["md5"] for s in C.SPLITS}:
            raise C.ProtocolError(f"La corrida {d} se entreno con otros datos que los del manifiesto.")
        tp = os.path.join(d, "test_metrics.json")
        if os.path.exists(tp):
            with open(tp, encoding="utf-8") as f:
                prev = json.load(f)
            if prev["test_md5"] != man["files"]["test"]["md5"]:
                raise C.ProtocolError(f"{tp} se calculo con otro test; no se sobrescribe.")
            print(f"  {os.path.basename(d):28s} ya evaluada, se mantiene")
            continue
        metas.append((d, m))

    ap_ = cfg["data"]["append_params"]
    bte_raw, t, yte = C.load_split(C.split_path(cfg, "test"), ap_, cfg["data"]["splits"]["test"]["n"])
    batch = cfg["train"]["val_infer_batch"]

    for d, m in metas:
        # La transformacion de entrada tambien sale de la metadata de la corrida
        bte = C.transform_branch(bte_raw, m["config"], m["architecture"]["branch"][0])
        net = C.build_net(m["architecture"], m["model"], m["sigma"]).to(device)
        ck = torch.load(os.path.join(d, "best.pt"), map_location=device)
        net.load_state_dict(ck["model_state_dict"])
        net.eval()
        sc = dict(np.load(os.path.join(d, "scalers.npz")))

        # Modo principal: referencia fija (determinista por muestra)
        net.unc_ref = m["unc_ref"]
        c, lo, up = C.predict(net, bte, t, sc, device, batch)
        c2, lo2, up2 = C.predict(net, bte, t, sc, device, 7)
        scale = C.pred_scale(c, lo, up)
        drift = max(np.abs(lo - lo2).max(), np.abs(up - up2).max(), np.abs(c - c2).max()) / scale
        if drift > 1e-4:
            raise C.ProtocolError(f"{d}: la prediccion cambia con el tamano de batch (desvio relativo {drift:.2e}).")
        main_m = C.metrics(yte, c, lo, up)
        if main_m["nonfinite"]:
            raise C.ProtocolError(f"{d}: metricas de test no finitas {main_m}")

        # Modo original (media del batch), solo como comparacion
        legacy = None
        if m["model"] in C.JAC_MODELS:
            net.unc_ref = None
            legacy = C.metrics(yte, *C.predict(net, bte, t, sc, device, LEGACY_BATCH))
            legacy["batch"] = LEGACY_BATCH

        C.write_json(os.path.join(d, "test_metrics.json"), {
            "commit_eval": commit, "commit_train": m["commit"], "test_md5": man["files"]["test"]["md5"],
            "batch_invariance_rel_drift": float(drift), "test": main_m, "test_legacy_batchmean": legacy})
        print(f"  {os.path.basename(d):28s} PICP={main_m['picp']:6.2f}%  MSE={main_m['mse']:.3e}  "
              f"IS={main_m['interval_score']:.3e}"
              + (f"  | legado PICP={legacy['picp']:.2f}%" if legacy else ""), flush=True)
    print(f"\nOK: {len(dirs)} corridas evaluadas en test.")


if __name__ == "__main__":
    main()
