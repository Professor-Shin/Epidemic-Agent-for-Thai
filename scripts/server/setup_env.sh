#!/bin/bash -l
# One-time setup on the login node: Python environment with the same package versions as the Colab runs.
#   bash scripts/server/setup_env.sh
source "$(dirname "$0")/common.sh"
cd "$PROJECT_DIR"
export TMPDIR="$PROJECT_DIR/.tmp"            # keep pip/singularity temp files off /tmp and /data/home (manual 6.1)
mkdir -p "$TMPDIR" "$HF_HOME"
load_modules

case "$ENV_KIND" in
  container)
    export SINGULARITY_CACHEDIR="$PROJECT_DIR/.singularity-cache" SINGULARITY_TMPDIR="$PROJECT_DIR/.singularity-tmp"
    mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$(dirname "$SIF")"
    [ -f "$SIF" ] || "$SINGULARITY_BIN" pull "$SIF" "$SIF_SOURCE"
    [ -d "$VENV" ] || "$SINGULARITY_BIN" exec --bind "$PROJECT_DIR" "$SIF" python3 -m venv "$VENV"
    "$SINGULARITY_BIN" cache clean --force >/dev/null 2>&1 || true ;;
  conda)
    set +u; eval "$CONDA_MODULE"; eval "$(conda shell.bash hook)"
    envs=$(conda env list)
    grep -qE "^${CONDA_ENV}[[:space:]]" <<< "$envs" || conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
    set -u ;;
  venv)
    [ -d "$VENV" ] || "python$PYTHON_VERSION" -m venv "$VENV" ;;
esac

activate_env
pip=(in_env python -m pip install --no-cache-dir)
"${pip[@]}" --upgrade pip
"${pip[@]}" "$TORCH_SPEC" --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
"${pip[@]}" -r requirements-server.txt
in_env python -c "import torch, transformers, bitsandbytes, papermill; print('python ok | torch', torch.__version__, '| transformers', transformers.__version__, '| bitsandbytes', bitsandbytes.__version__)"
echo "Environment ready. Next: bash scripts/server/download.sh"
