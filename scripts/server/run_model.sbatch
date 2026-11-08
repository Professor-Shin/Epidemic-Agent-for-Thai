#!/bin/bash -l
#SBATCH --job-name=gabm-model
#SBATCH --output=logs/%x_%A_%a.out
# One array task = one model from MODELS: runs STAGES in order (default: feasibility -> C -> D).
# Partition, QoS, GPU, time and memory come from server.env through submit_all.sh; do not sbatch this file directly.
: "${PROJECT_DIR:?submit with: bash scripts/server/submit_all.sh}"
source "$PROJECT_DIR/scripts/server/common.sh"
activate_env
cd "$PROJECT_DIR"

read -ra MODEL_LIST <<< "$MODELS"
MODEL="${MODEL_LIST[$SLURM_ARRAY_TASK_ID]}"
# SEEDS_OVERRIDE and FORCE are for one-off supplementary runs, e.g. a longer horizon where condition C does
# not exist yet at that DAYS and the D gate would therefore refuse. Leave both unset for the panel.
SEEDS="${SEEDS_OVERRIDE:-$(in_env python -c 'from gabm import load_config; print(*load_config()["sim"]["seeds"])')}"
run() { in_env python scripts/run_server.py --model "$MODEL" --days "$DAYS" ${FORCE:+--force} "$@"; }

echo "[$(date '+%F %T')] model=$MODEL days=$DAYS stages='$STAGES' seeds='$SEEDS' node=$(hostname)"
run --stage check

for stage in $STAGES; do
  echo "[$(date '+%F %T')] stage $stage"
  case "$stage" in
    feasibility)
      run --stage feasibility || echo "feasibility failed for $MODEL (not needed for C/D, continuing)" ;;
    C|D)
      if [ "$SEEDS_IN_PARALLEL" = 1 ]; then
        pids=()
        for s in $SEEDS; do run --stage "$stage" --seeds "$s" & pids+=($!); done
        fail=0
        for p in "${pids[@]}"; do wait "$p" || fail=1; done
        if [ "$fail" = 1 ]; then echo "stage $stage failed for $MODEL" >&2; exit 1; fi
      else
        run --stage "$stage"
      fi ;;
    sens)
      run --stage sens ${SENS_AXES:+--axes "$SENS_AXES"} ;;
    order|baseline)
      run --stage "$stage" ;;
    *)
      echo "unknown stage '$stage' in STAGES" >&2; exit 1 ;;
  esac
done
echo "[$(date '+%F %T')] $MODEL done"
