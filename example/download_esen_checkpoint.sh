#!/usr/bin/env bash
set -euo pipefail

# This helper does not bypass the gated model license. The Hugging Face account
# owning HF_TOKEN must first accept the facebook/OMAT24 terms in a browser.

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/checkpoints"}

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set." >&2
    echo "Accept the facebook/OMAT24 terms, create a read token, then export HF_TOKEN." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="facebook/OMAT24",
    filename="esen_30m_oam.pt",
    token=os.environ["HF_TOKEN"],
    local_dir=os.environ["OUTPUT_DIR"],
)
print(path)
PY
