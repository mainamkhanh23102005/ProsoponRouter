#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/gemma-4-E2B-it-Q4_K_M.gguf}"

if [[ "${LOCAL_LLM:-0}" != "0" && -f "$MODEL_PATH" && -x "$(command -v llama-server)" ]]; then
  llama-server -m "$MODEL_PATH" \
    -c "${LOCAL_LLM_CTX:-2048}" \
    --threads "${LOCAL_LLM_THREADS:-2}" \
    --host 127.0.0.1 --port 8080 \
    --no-webui \
    --reasoning-budget 0 \
    >/tmp/llama-server.log 2>&1 &
  echo "local_llm_start pid=$! model=$MODEL_PATH" >&2
  wait_seconds="${LOCAL_LLM_WAIT_READY:-0}"
  waited=0
  while [[ "$waited" -lt "$wait_seconds" ]]; do
    if python - <<'PY'
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
except Exception:
    raise SystemExit(1)
PY
    then
      echo "local_llm_ready_after=${waited}s" >&2
      break
    fi
    sleep 2
    waited=$((waited + 2))
  done
else
  echo "local_llm_disabled_or_missing" >&2
fi

python -m src.main
