"""
Genera el indice de modelos, datos y configuraciones del protocolo.

    python protocol/index_models.py > /tmp/MODELS.md      (en el cluster)
    scp hpc:/tmp/MODELS.md repositorio_tesis/protocol/   (desde local, y commitear)

Solo lee metadatos: no necesita GPU ni Slurm y no modifica nada. Esta pensado
para regenerarse cada vez que se agregan corridas, en vez de mantener a mano un
documento que se desactualiza en cuanto se lanza el siguiente barrido.

Escribir la salida FUERA del repo del cluster: si se sobrescribe
protocol/MODELS.md alli, el working tree queda modificado y el siguiente
git pull choca ("local changes would be overwritten"). El commit se hace
siempre desde local.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CLUSTER = "/nfs_home/rprieto/Tesis-RP"
CONFIGS = sorted(os.path.basename(p)[:-5]
                 for p in glob.glob(os.path.join(C.REPO, "protocol", "configs", "*.json")))


def leer(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rel(path):
    return os.path.relpath(path, C.REPO).replace("\\", "/")


def configs():
    for b in CONFIGS:
        try:
            yield b, C.load_config(b)
        except C.ProtocolError as e:
            print(f"<!-- {b}: no se pudo cargar ({e}) -->")


def resultados(d):
    """Metricas de test y correlaciones de una corrida, si ya fue evaluada."""
    out = {"ood": sorted(os.path.basename(p)[4:-5] for p in glob.glob(os.path.join(d, "ood_*.json")))}
    p = os.path.join(d, "test_metrics.json")
    if os.path.exists(p):
        out.update(leer(p)["test"])
    p = os.path.join(d, "test_correlations.json")
    if os.path.exists(p):
        j = leer(p)
        out.update({k: j[k] for k in ("rho_err_spearman", "rho_sens_spearman") if k in j})
        if "params" in j:
            out["params"] = int(j["params"])
    return out


def num(d, k, fmt="{:.3e}"):
    return fmt.format(d[k]) if d.get(k) is not None else "-"


def seccion_datos():
    print("## Datos\n")
    print("Un solo conjunto por benchmark: las variantes de modelo (`data_from`) reusan los "
          "mismos archivos, verificados por md5 contra `MANIFEST.json`.\n")
    print("| Benchmark de datos | Split | Ruta | n | Semilla | md5 / overrides |")
    print("|---|---|---|---|---|---|")
    vistos = set()
    for _, cfg in configs():
        dd = cfg["_data_benchmark"]
        if dd in vistos:
            continue
        vistos.add(dd)
        man_p = C.manifest_path(cfg)
        man = leer(man_p) if os.path.exists(man_p) else None
        for s in C.SPLITS:
            sp = cfg["data"]["splits"][s]
            md5 = f"`{man['files'][s]['md5'][:12]}`" if man else "*sin manifiesto*"
            print(f"| {dd} | {s} | `{rel(C.split_path(cfg, s))}` | {sp['n']} | "
                  f"{sp.get('seed', '-')} | {md5} |")
        for name, o in sorted(cfg.get("ood", {}).items()):
            falta = "" if os.path.exists(C.ood_path(cfg, name)) else " *(sin generar)*"
            gen = ", ".join(f"{k}={v}" for k, v in o.get("gen_overrides", {}).items())
            print(f"| {dd} | ood:{name}{falta} | `{rel(C.ood_path(cfg, name))}` | {o['n']} | "
                  f"{o['seed']} | {gen or '-'} |")


def seccion_modelos():
    print("\n## Modelos entrenados\n")
    print("Cada corrida es una carpeta con `best.pt` (checkpoint elegido por validacion), "
          "`scalers.npz`, `metadata.json`, `val_history.csv`, `loss_history.csv` y, si ya fue "
          "evaluada, `test_metrics.json` / `test_correlations.json` / `ood_<nombre>.json`.\n")
    print("Una corrida marcada *(running)* sin job vivo en `squeue` fue interrumpida "
          "(cancelada o caida): su carpeta esta incompleta y no se debe usar.\n")
    for b, cfg in configs():
        dirs = [d for d in sorted(glob.glob(os.path.join(C.runs_dir(cfg), "*_seed*")))
                if os.path.exists(os.path.join(d, "metadata.json"))]
        if not dirs:
            continue
        br = cfg["model"]["branch"]
        tr = cfg.get("input_transform")
        entrada = (f", entrada {tr['grid'][0] // tr['stride']}x{tr['grid'][1] // tr['stride']}"
                   if tr else "")
        print(f"### `{b}` - branch {br[0]} -> {len(br) - 1}x{br[1]}{entrada}, "
              f"datos de `{cfg['_data_benchmark']}`\n")
        print(f"Ruta base: `{rel(C.runs_dir(cfg))}/`\n")
        print("| Corrida | Modelo | lambda | Semilla | Params | Paso elegido | MSE test | PICP | MPIW | "
              "IS | rhoErr | rhoSens | OOD |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for d in dirs:
            m = leer(os.path.join(d, "metadata.json"))
            r = resultados(d)
            nombre = os.path.basename(d)
            if m["status"] != "completed":
                nombre += f" *({m['status']})*"
            paso = f"{m.get('best_step', '-')}/{m.get('iterations_run', '-')}"
            print(f"| `{nombre}` | {m['model']} | {m['lambda']:g} | {m['seed']} | "
                  f"{num(r, 'params', '{:,}')} | {paso} | "
                  f"{num(r, 'mse')} | {num(r, 'picp', '{:.2f}%')} | {num(r, 'mpiw', '{:.4f}')} | "
                  f"{num(r, 'interval_score', '{:.4f}')} | {num(r, 'rho_err_spearman', '{:.3f}')} | "
                  f"{num(r, 'rho_sens_spearman', '{:.3f}')} | {', '.join(r['ood']) or '-'} |")
        print()


def seccion_configs():
    print("## Configuraciones\n")
    print("Fuente de verdad: `protocol/configs/<benchmark>.json`. Cada corrida guarda ademas "
          "una copia literal de la config con la que se entreno en su `metadata.json`.\n")
    print("| Config | Modelos del barrido | lambda | Semillas | n_train | Arquitectura | sigma | "
          "Entrada | Optimizador |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b, cfg in configs():
        t, mo, sw = cfg["train"], cfg["model"], cfg["sweep"]
        sig = ", ".join(f"{k}={v}" for k, v in mo["sigma"].items() if k in sw["models"]) or "-"
        tr = cfg.get("input_transform")
        ent = (f"{tr['grid'][0]}x{tr['grid'][1]} stride {tr['stride']}" if tr
               else f"{mo['branch'][0]} directo")
        print(f"| `{b}` | {', '.join(sw['models'])} | {', '.join(f'{x:g}' for x in sw['lambdas'])} | "
              f"{', '.join(str(x) for x in sw['seeds'])} | {cfg['data']['splits']['train']['n']} | "
              f"{len(mo['branch']) - 1}x{mo['branch'][1]}, K={mo['trunk'][-1]}, {mo['activation']} | "
              f"{sig} | {ent} | lr {t['learning_rate']}, decay {t['decay_rate']}/{t['decay_steps']}, "
              f"batch {t['batch_size']}, {t['iterations']} it, patience {t.get('patience', '-')} |")
    print("\nQue es cada config:\n")
    for b, cfg in configs():
        print(f"- **`{b}`** - {cfg.get('description', '')}")


def seccion_legado():
    subs = sorted(p for p in glob.glob(os.path.join(C.REPO, "modelos", "*")) if os.path.isdir(p))
    if not subs:
        return
    print("\n## Modelos anteriores al protocolo\n")
    print("Entrenados con `scripts/*/*_sweep.sh` sobre los datos con leakage de semillas. "
          "Se conservan como referencia historica de los resultados del paper; "
          "**no usar para resultados nuevos**.\n")
    print("| Carpeta | Checkpoints `.pt` |")
    print("|---|---|")
    for sub in subs:
        n = len(glob.glob(os.path.join(sub, "**", "*.pt"), recursive=True))
        if n:
            print(f"| `modelos/{os.path.basename(sub)}/` | {n} |")


def main():
    print("# Modelos, datos y configuraciones del protocolo\n")
    print("**Archivo generado.** Regenerar con `python protocol/index_models.py > protocol/MODELS.md` "
          "despues de cada barrido; no editar a mano.\n")
    print(f"Las rutas son relativas a la raiz del repo: `{CLUSTER}` en el cluster (host `hpc`), "
          "`repositorio_tesis/` en local. Los `runs/` y los `data/` viven solo en el cluster.\n")
    seccion_datos()
    seccion_modelos()
    seccion_configs()
    seccion_legado()
    print("\n## Cargar una corrida\n")
    print("```python")
    print("import json, sys, numpy as np, torch; sys.path.insert(0, 'protocol')")
    print("import common as C")
    print("d = 'runs/protocol/ns2d_n20k/jacobian_lam4_seed0'")
    print("m = json.load(open(d + '/metadata.json'))")
    print("net = C.build_net(m['architecture'], m['model'], m['sigma']).cuda()")
    print("net.load_state_dict(torch.load(d + '/best.pt')['model_state_dict']); net.eval()")
    print("net.unc_ref = m['unc_ref']        # referencia fija: el intervalo no depende del batch")
    print("sc = dict(np.load(d + '/scalers.npz')); cfg = m['config']")
    print("```")
    print("\nLa arquitectura, el sigma, los scalers y la transformacion de entrada salen de "
          "`metadata.json`, nunca de la linea de comandos: por construccion es imposible evaluar "
          "una corrida con hiperparametros distintos a los de su entrenamiento.")


if __name__ == "__main__":
    main()
