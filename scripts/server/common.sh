# Sourced by every script in scripts/server: loads server.env and defines the environment helpers.
# Usage in a script:  activate_env   then   in_env python ...   (in_env runs inside the container when ENV_KIND=container)
set -euo pipefail
_SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$_SERVER_DIR/server.env"
set +a
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE="$HF_OFFLINE" TRANSFORMERS_OFFLINE="$HF_OFFLINE"

# module and conda shell functions read unset variables, so `set -u` is off while they run
load_modules() {
  if [ -n "$MODULE_LOAD" ]; then set +u; eval "$MODULE_LOAD"; set -u; fi
}

activate_env() {
  load_modules
  set +u
  case "$ENV_KIND" in
    container) ;;                                   # nothing to activate: in_env wraps each command
    conda) eval "$CONDA_MODULE"; eval "$(conda shell.bash hook)"; conda activate "$CONDA_ENV" ;;
    venv) source "$VENV/bin/activate" ;;
    *) echo "unknown ENV_KIND '$ENV_KIND' in server.env" >&2; return 1 ;;
  esac
  set -u
}

in_env() {
  if [ "$ENV_KIND" != container ]; then "$@"; return; fi
  local opts=(--bind "$PROJECT_DIR")
  case "$HF_HOME" in "$PROJECT_DIR"/*) ;; *) opts+=(--bind "$HF_HOME") ;; esac
  if command -v nvidia-smi >/dev/null 2>&1; then opts+=(--nv); fi   # GPU nodes only
  "$SINGULARITY_BIN" exec "${opts[@]}" "$SIF" \
    env PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" "$@"
}
