#!/bin/bash -l
# Run on the login node (it has internet): downloads the models in MODELS into $HF_HOME.
#   bash -l scripts/server/download.sh
# llama8b is gated: accept its licence on huggingface.co and put your token in $HF_HOME/token (default
# .hf_cache/token, gitignored), or export HF_TOKEN. The other models need no token.
source "$(dirname "$0")/common.sh"
activate_env
cd "$PROJECT_DIR"
mkdir -p "$HF_HOME"
# shellcheck disable=SC2086  # MODELS is a space-separated list
in_env python scripts/download_models.py $MODELS     # the script switches HF_HUB_OFFLINE off for itself
du -sh "$HF_HOME" 2>/dev/null || true
echo "Models ready. Next: bash scripts/server/submit_all.sh smoke"
