#!/bin/bash
# ==============================================================================
# Lanza el protocolo completo de un benchmark como una cadena de jobs de Slurm:
#
#   gen (train, val, test en paralelo) -> verify -> train (job array) -> eval -> aggregate
#
# Cada eslabon depende del anterior con afterok: si uno falla, la cadena se
# detiene y nada posterior se ejecuta.
#
# Uso (desde la raiz del repo, en el nodo de login):
#   bash protocol/slurm/launch.sh <benchmark> [--smoke] [--from gen|train|eval|aggregate]
#                                 [--train-gres gpu:a100:1] [--eval-gres gpu:a30mig:1]
#                                 [--after JOBID]   (encadena la primera etapa a un job ya encolado)
# ==============================================================================
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BENCH=${1:?Uso: launch.sh <benchmark> [opciones]}; shift
SMOKE=""; FROM="gen"; AFTER=""; TRAIN_GRES="gpu:a100:1"; EVAL_GRES="gpu:a30mig:1"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="--smoke" ;;
    --from) FROM=$2; shift ;;
    --train-gres) TRAIN_GRES=$2; shift ;;
    --eval-gres) EVAL_GRES=$2; shift ;;
    --after) AFTER=$2; shift ;;
    *) echo "Argumento desconocido: $1" >&2; exit 2 ;;
  esac
  shift
done

# Guardia temprana (cada job la vuelve a aplicar): nada desde codigo sin commitear
# (protocol/MODELS.md es un indice generado: se excluye, no determina resultados)
CODE_PATHS=(src deepxde-extensions protocol ':(exclude)protocol/MODELS.md')
if [ -n "$(git status --porcelain -- "${CODE_PATHS[@]}")" ]; then
  echo "[PROTOCOLO VIOLADO] Hay cambios sin commitear en src/, deepxde-extensions/ o protocol/." >&2
  git status --short -- "${CODE_PATHS[@]}" >&2
  exit 3
fi

mkdir -p logs/protocol
source env_tesis/bin/activate
N=$(python protocol/plan.py "$BENCH" $SMOKE --count)
echo "Benchmark: $BENCH $SMOKE | corridas en el barrido: $N | commit $(git rev-parse --short HEAD)"

DEP=""
[ -n "$AFTER" ] && DEP="--dependency=afterok:$AFTER"
if [ "$FROM" = "gen" ]; then
  G=""
  for s in train val test; do
    j=$(sbatch --parsable $DEP -J "proto_gen_${BENCH}_$s" protocol/slurm/gen.sh "$BENCH" "$s" $SMOKE)
    G="$G:$j"; echo "  gen $s: $j"
  done
  V=$(sbatch --parsable --dependency=afterok$G -J "proto_verify_$BENCH" protocol/slurm/verify.sh "$BENCH" $SMOKE)
  echo "  verify:  $V"; DEP="--dependency=afterok:$V"; FROM="train"
fi
if [ "$FROM" = "train" ]; then
  T=$(sbatch --parsable $DEP --array=0-$((N - 1)) --gres="$TRAIN_GRES" -J "proto_train_$BENCH" \
      protocol/slurm/train.sh "$BENCH" $SMOKE)
  echo "  train:   $T (array 0-$((N - 1)), $TRAIN_GRES)"; DEP="--dependency=afterok:$T"; FROM="eval"
fi
if [ "$FROM" = "eval" ]; then
  E=$(sbatch --parsable $DEP --gres="$EVAL_GRES" -J "proto_eval_$BENCH" protocol/slurm/eval.sh "$BENCH" $SMOKE)
  echo "  eval:    $E ($EVAL_GRES)"; DEP="--dependency=afterok:$E"; FROM="aggregate"
fi
if [ "$FROM" = "aggregate" ]; then
  A=$(sbatch --parsable $DEP -J "proto_agg_$BENCH" protocol/slurm/aggregate.sh "$BENCH" $SMOKE)
  echo "  tabla:   $A"
fi
