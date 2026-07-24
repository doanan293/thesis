#!/usr/bin/env sh
set -eu

: "${LLAMA_MODEL:?LLAMA_MODEL is required}"
LLAMA_ROLE_PROTOCOL="${LLAMA_ROLE_PROTOCOL:-embedding}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
LLAMA_CONTEXT_SIZE="${LLAMA_CONTEXT_SIZE:-2048}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-2048}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-512}"

set -- \
  --model "$LLAMA_MODEL" \
  --host 0.0.0.0 \
  --port 8080 \
  --no-webui \
  --offline \
  -np "$LLAMA_PARALLEL" \
  -c "$LLAMA_CONTEXT_SIZE" \
  -b "$LLAMA_BATCH_SIZE" \
  -ub "$LLAMA_UBATCH_SIZE"

case "$LLAMA_ROLE_PROTOCOL" in
  embedding)
    set -- "$@" --embedding
    ;;
  native_rerank)
    set -- "$@" --embedding --pooling rank
    ;;
  completion_logprobs)
    ;;
  *)
    echo "Unsupported LLAMA_ROLE_PROTOCOL: $LLAMA_ROLE_PROTOCOL" >&2
    exit 2
    ;;
esac

exec /app/llama-server "$@"
