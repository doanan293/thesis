#!/usr/bin/env sh
# llama-server launcher for the CPU-only compose services. Every value comes from the
# environment, so a lighter GGUF file or different CPU tuning needs no change here.
set -eu

: "${LLAMA_MODEL:?LLAMA_MODEL is required}"
LLAMA_ROLE_PROTOCOL="${LLAMA_ROLE_PROTOCOL:-embedding}"
LLAMA_PARALLEL="${LLAMA_PARALLEL:-1}"
# llama-server divides -c across slots, so the total context is per-slot context x slots.
LLAMA_CONTEXT_PER_SLOT="${LLAMA_CONTEXT_PER_SLOT:-2048}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-$LLAMA_CONTEXT_PER_SLOT}"
LLAMA_UBATCH_SIZE="${LLAMA_UBATCH_SIZE:-$LLAMA_BATCH_SIZE}"
# Empty means llama-server picks the thread count itself.
LLAMA_THREADS="${LLAMA_THREADS:-}"
LLAMA_THREADS_BATCH="${LLAMA_THREADS_BATCH:-$LLAMA_THREADS}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-auto}"
# Same cache policy as corpus-pipeline's runtime (server_policy.inference_cache_policy).
LLAMA_CACHE_RAM_MIB="${LLAMA_CACHE_RAM_MIB:-0}"
LLAMA_SLOT_PROMPT_SIMILARITY="${LLAMA_SLOT_PROMPT_SIMILARITY:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"

set -- \
  --model "$LLAMA_MODEL" \
  --host 0.0.0.0 \
  --port 8080 \
  --no-webui \
  --offline \
  -np "$LLAMA_PARALLEL" \
  -c "$((LLAMA_CONTEXT_PER_SLOT * LLAMA_PARALLEL))" \
  -b "$LLAMA_BATCH_SIZE" \
  -ub "$LLAMA_UBATCH_SIZE" \
  -fa "$LLAMA_FLASH_ATTN" \
  --cache-ram "$LLAMA_CACHE_RAM_MIB" \
  --no-cache-idle-slots

if [ -n "$LLAMA_THREADS" ]; then
  set -- "$@" -t "$LLAMA_THREADS"
fi
if [ -n "$LLAMA_THREADS_BATCH" ]; then
  set -- "$@" -tb "$LLAMA_THREADS_BATCH"
fi
if [ -n "$LLAMA_SLOT_PROMPT_SIMILARITY" ]; then
  set -- "$@" --slot-prompt-similarity "$LLAMA_SLOT_PROMPT_SIMILARITY"
fi

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

# Intentionally unquoted: LLAMA_EXTRA_ARGS holds several flags separated by spaces.
# shellcheck disable=SC2086
set -- "$@" $LLAMA_EXTRA_ARGS

echo "llama-server $*"
exec /app/llama-server "$@"
