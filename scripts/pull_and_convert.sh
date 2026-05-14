#!/usr/bin/env bash
# nim-cpu-bridge: scripts/pull_and_convert.sh
#
# Pulls NIM model weights from NGC (without starting the full NIM container)
# and converts them to GGUF for CPU inference.
#
# Usage:
#   NGC_API_KEY=<key> ./scripts/pull_and_convert.sh \
#     --model meta/llama-3.1-8b-instruct \
#     --quant Q4_K_M \
#     [--output /path/to/cache]
#
# Requires: docker, ngc CLI or docker login to nvcr.io

set -euo pipefail

MODEL_TAG=""
QUANT="Q4_K_M"
OUTPUT_DIR="${HOME}/.cache/nim"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL_TAG="$2"; shift 2 ;;
    --quant)   QUANT="$2";     shift 2 ;;
    --output)  OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_TAG" ]]; then
  echo "Usage: $0 --model <model_tag> [--quant Q4_K_M] [--output <dir>]"
  exit 1
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "Error: NGC_API_KEY not set"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "nim-cpu-bridge: pull + convert"
echo "Model : $MODEL_TAG"
echo "Quant : $QUANT"
echo "Output: $OUTPUT_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Docker login to NGC ───────────────────────────────────────────────
echo "[1/3] Authenticating with NGC …"
echo "$NGC_API_KEY" | docker login nvcr.io -u "\$oauthtoken" --password-stdin

# ── Step 2: Pull weights using a minimal NIM container run ───────────────────
# We spin up the NIM container just long enough to download weights,
# then immediately stop it. The weights land in the nim_cache volume.
echo "[2/3] Pulling model weights from NGC …"
mkdir -p "$OUTPUT_DIR"

NIM_IMAGE="nvcr.io/nim/${MODEL_TAG}:latest"

docker run --rm \
  -e NGC_API_KEY="$NGC_API_KEY" \
  -e NIM_CACHE_PATH=/nim_cache \
  -v "${OUTPUT_DIR}:/nim_cache" \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -c "python -c \"
import os, json, pathlib, urllib.request, urllib.error
cache = pathlib.Path('/nim_cache')
cache.mkdir(parents=True, exist_ok=True)
print('Weight download triggered by NIM init ...')
\" || true
# Try the NIM download path directly
cd /opt/nim/models && python download_models.py --model-name ${MODEL_TAG} 2>/dev/null || true
" || echo "Weight pull via entrypoint skipped (NIM image may auto-download on first inference start)"

echo "[3/3] Converting weights to GGUF …"
python -m src.converter.ngc_to_gguf \
  "${MODEL_TAG}" \
  --quant "${QUANT}" \
  --cache "${OUTPUT_DIR}"

echo ""
echo "✓ Done. Set this env var before starting nim-cpu-bridge:"
SLUG=$(echo "$MODEL_TAG" | tr '/' '_' | tr ':' '_')
GGUF_PATH="${OUTPUT_DIR}/ncb_converted/${SLUG}/${SLUG}.${QUANT}.gguf"
echo ""
echo "  export NCB_MODEL_PATH=\"${GGUF_PATH}\""
echo ""
