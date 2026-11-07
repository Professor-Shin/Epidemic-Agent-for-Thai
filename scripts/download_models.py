"""Download model weights into the Hugging Face cache ($HF_HOME) before submitting jobs.

Run it on the login node (it has internet; compute nodes may not), e.g. via scripts/server/download.sh:

  python scripts/download_models.py llama8b typhoon8b qwen7b openthaigpt7b seallm7b

Only the files transformers needs are fetched: config, tokenizer and one copy of the weights (safetensors when
the repo has them). Gated models (llama8b) need the licence accepted on huggingface.co and a token, either in
$HF_HOME/token or in HF_TOKEN (huggingface_hub reads both).
"""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"     # set before importing huggingface_hub: downloading needs the network

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, constants, snapshot_download  # noqa: E402

from gabm import load_config  # noqa: E402

KEEP = ["*.json", "*.model", "*.tiktoken", "*.txt", "tokenizer*"]
SKIP = ["original/*", "consolidated*"]   # Meta / Mistral native-format duplicates of the weights


def main():
    cfg = load_config(ROOT / "config" / "base.yaml")
    keys = sys.argv[1:] or cfg["llm"]["compare"]
    unknown = [k for k in keys if k not in cfg["llm"]["models"]]
    if unknown:
        sys.exit(f"unknown model keys {unknown}; choose from {list(cfg['llm']['models'])}")
    token = None    # huggingface_hub then uses HF_TOKEN or $HF_HOME/token
    api = HfApi(token=token)
    print(f"cache: {constants.HF_HUB_CACHE}")
    failed = []
    for k in keys:
        repo = cfg["llm"]["models"][k]["hf_id"]
        try:
            files = api.list_repo_files(repo)
            weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
            path = snapshot_download(repo, allow_patterns=KEEP + weights, ignore_patterns=SKIP, token=token)
            print(f"ok   {k:<14} {repo} -> {path}")
        except Exception as e:
            hint = (" (gated: accept the licence on huggingface.co and put your token in $HF_HOME/token)"
                    if cfg["llm"]["models"][k].get("gated") else "")
            print(f"FAIL {k:<14} {repo}: {type(e).__name__}: {e}{hint}")
            failed.append(k)
    if failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
