#!/bin/bash
# Submit the runs to SLURM from the login node. Settings come from server.env.
#   bash scripts/server/submit_all.sh smoke     # 2-day test of the first model: env, GPU, cached weights (~10 min)
#   bash scripts/server/submit_all.sh           # full run: one GPU job per model, then the analysis job
#   bash scripts/server/submit_all.sh analyze   # analysis only (e.g. after resubmitting a timed-out model)
source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"
mkdir -p logs

gpu=(--export=ALL --partition="$PARTITION" --qos="$QOS" --gres="$GPU_GRES" --cpus-per-task="$CPUS_PER_TASK" --mem="$MEM")
cpu=(--export=ALL --qos="$CPU_QOS" --cpus-per-task=4 --mem="$MEM_CPU" --time="$TIME_CPU")
if [ -n "$CPU_PARTITION" ]; then cpu+=(--partition="$CPU_PARTITION"); fi

read -ra MODEL_LIST <<< "$MODELS"
n=${#MODEL_LIST[@]}

case "${1:-all}" in
  smoke)
    id=$(sbatch --parsable "${gpu[@]}" --time=00:45:00 scripts/server/smoke.sbatch)
    echo "smoke test $id (${MODEL_LIST[0]}). Watch: tail -f logs/gabm-smoke_$id.out" ;;
  analyze)
    id=$(sbatch --parsable "${cpu[@]}" scripts/server/analyze.sbatch)
    echo "analysis $id. Watch: tail -f logs/gabm-analyze_$id.out" ;;
  all)
    arr=$(sbatch --parsable "${gpu[@]}" --time="$TIME_GPU" --array="0-$((n - 1))%$MAX_PARALLEL" \
          scripts/server/run_model.sbatch)
    ana=$(sbatch --parsable "${cpu[@]}" --dependency="afterany:$arr" scripts/server/analyze.sbatch)
    echo "model array $arr: $n models ($MODELS), DAYS=$DAYS, stages: $STAGES, at most $MAX_PARALLEL at once"
    echo "analysis $ana starts when the array ends"
    echo "Watch: squeue -u $USER   |   tail -f logs/gabm-model_${arr}_0.out" ;;
  *)
    echo "usage: $0 [smoke|all|analyze]" >&2; exit 1 ;;
esac
