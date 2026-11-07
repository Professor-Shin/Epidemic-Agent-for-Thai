#!/bin/bash -l
#SBATCH --job-name=gabm-analyze
#SBATCH --output=logs/%x_%j.out
# Runs the analysis cells of notebooks 04 and 05 (RUN_LLM=False, no GPU) on everything in outputs/runs.
# The executed notebooks (tables + figures inline) go to outputs/executed/, figures to outputs/figures/,
# and notebook 04 writes outputs/metrics.csv. submit_all.sh starts this job after the model array ends.
: "${PROJECT_DIR:?submit with: bash scripts/server/submit_all.sh}"
source "$PROJECT_DIR/scripts/server/common.sh"
activate_env
cd "$PROJECT_DIR"
mkdir -p outputs/executed

pm() { in_env papermill -k "$PAPERMILL_KERNEL" --cwd notebooks "$@"; }

pm notebooks/04_model_comparison.ipynb outputs/executed/04_model_comparison.ipynb \
   -p RUN_LLM False -p DAYS "$DAYS" || echo "notebook 04 analysis failed (see the executed notebook)" >&2
for m in $MODELS; do
  pm notebooks/05_leader_agent.ipynb "outputs/executed/05_leader_$m.ipynb" \
     -p MODEL_KEY "$m" -p RUN_LLM False -p DAYS "$DAYS" || echo "notebook 05 analysis failed for $m" >&2
done
echo "analysis done: outputs/executed/, outputs/figures/, outputs/metrics.csv"
