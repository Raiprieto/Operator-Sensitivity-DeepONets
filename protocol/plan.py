"""
Lista las corridas declaradas en el barrido de un benchmark.

    python protocol/plan.py <benchmark> [--smoke] [--count]

El indice de cada corrida es el SLURM_ARRAY_TASK_ID que usa protocol/slurm/train.sh.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("benchmark")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--count", action="store_true")
a = ap.parse_args()
runs = C.plan(C.load_config(a.benchmark, a.smoke))
if a.count:
    print(len(runs))
else:
    for i, (m, lam, s) in enumerate(runs):
        print(f"{i:3d}  {C.run_name(m, lam, s)}")
