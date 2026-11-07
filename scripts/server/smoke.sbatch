#!/bin/bash -l
#SBATCH --job-name=gabm-smoke
#SBATCH --output=logs/%x_%j.out
# Quick check before the full run: env + GPU + cached weights, then a 2-day condition C run (tag "smoke",
# left out of every analysis) with the first model in MODELS.
: "${PROJECT_DIR:?submit with: bash scripts/server/submit_all.sh smoke}"
source "$PROJECT_DIR/scripts/server/common.sh"
activate_env
cd "$PROJECT_DIR"

read -ra MODEL_LIST <<< "$MODELS"
m="${MODEL_LIST[0]}"
nvidia-smi || true
in_env python scripts/run_server.py --model "$m" --stage check
in_env python scripts/run_server.py --model "$m" --stage smoke
echo "smoke test OK for $m: submit the full run with  bash scripts/server/submit_all.sh"
