"""
Genera un split (train | val | test) de un benchmark segun su configuracion.

    python protocol/generate.py <benchmark> <split> [--smoke]

Usa los generadores existentes por linea de comandos, con la semilla base
registrada en protocol/configs/<benchmark>.json. Escribe a un archivo temporal
y solo lo renombra al final, asi un job interrumpido nunca deja un split a medias.
"""

import argparse
import os
import subprocess
import sys

import h5py

sys.path.insert(0, os.path.dirname(__file__))
from common import (REPO, ProtocolError, load_config, ood_path, require_absent,  # noqa: E402
                    require_clean_code, require_slurm, split_path, write_json)

GEN = os.path.join(REPO, "src", "data_generation")


def command(cfg, split, out):
    s = cfg["ood"][split[4:]] if split.startswith("ood:") else cfg["data"]["splits"][split]
    ga = cfg["data"]["generator_args"]
    g, n, seed = s["generator"], str(s["n"]), str(s["seed"])
    py = [sys.executable, "-u"]
    if g == "burgers":
        cmd = py + [f"{GEN}/burgers_pdebench.py", "--num_samples", n, "--nx", str(ga["nx"]),
                    "--nt", str(ga["nt"]), "--L", str(ga["L"]), "--T", str(ga["T"]),
                    "--nu_min", str(ga["nu_min"]), "--nu_max", str(ga["nu_max"]),
                    "--batch_size", str(ga["batch_size"]), "--seed_offset", seed]
    elif g == "ns2d":
        cmd = py + [f"{GEN}/navier_stokes_state_pdebench.py", "--num_samples", n,
                    "--nx", str(ga["nx"]), "--T", str(ga["T"]),
                    "--batch_size", str(ga["batch_size"]), "--seed_offset", seed]
    elif g == "darcy":
        cmd = py + [f"{GEN}/darcy_pdebench.py", "--num_samples", n, "--N", str(ga["N"]),
                    "--batch_size", str(ga["batch_size"]), "--seed_offset", seed]
    elif g == "biharmonic_scaled":
        cmd = py + [f"{GEN}/biharmonic_scaled_test.py", "--train_path", split_path(cfg, "train"),
                    "--num_samples", n, "--seed", seed,
                    "--c_min", str(ga["c_min"]), "--c_max", str(ga["c_max"])]
    else:
        raise ProtocolError(f"Generador desconocido: {g}")
    if g != "biharmonic_scaled" and not s.get("jacobian", False):
        cmd.append("--skip_jacobian")
    for k, v in s.get("gen_overrides", {}).items():   # p. ej. rangos desplazados para OOD
        cmd += [f"--{k}", str(v)]
    return cmd + ["--output_path", out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("split", help="train | val | test | ood:<nombre>")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    require_slurm()
    commit = require_clean_code()
    cfg = load_config(a.benchmark, a.smoke)
    es_ood = a.split.startswith("ood:")
    s = cfg["ood"][a.split[4:]] if es_ood else cfg["data"]["splits"][a.split]
    if "file" in s:
        print(f"'{a.split}' de {a.benchmark} es un archivo fijo ({s['file']}); no se genera.")
        return

    out = ood_path(cfg, a.split[4:]) if es_ood else split_path(cfg, a.split)
    require_absent(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".partial.h5"
    if os.path.exists(tmp):
        os.remove(tmp)     # resto de un intento anterior interrumpido: nunca se uso

    cmd = command(cfg, a.split, tmp)
    print("$ " + " ".join(cmd), flush=True)
    env = dict(os.environ, JAX_PLATFORM_NAME=os.environ.get("JAX_PLATFORM_NAME", "cpu"))
    r = subprocess.run(cmd, cwd=REPO, env=env)
    if r.returncode != 0:
        raise ProtocolError(f"El generador fallo con codigo {r.returncode}")

    with h5py.File(tmp, "r") as f:
        got = f["branch_inputs"].shape[0]
    if got != s["n"]:
        raise ProtocolError(f"Se esperaban {s['n']} muestras y el generador escribio {got}")
    os.replace(tmp, out)
    write_json(out + ".provenance.json", {"benchmark": a.benchmark, "split": a.split, "commit": commit,
                                          "command": cmd[1:], "split_config": s,
                                          "generator_args": cfg["data"]["generator_args"],
                                          "slurm_job": os.environ.get("SLURM_JOB_ID")})
    print(f"OK: {out} ({got} muestras)")


if __name__ == "__main__":
    main()
