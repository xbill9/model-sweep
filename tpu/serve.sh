#!/bin/bash
# serve.sh <model> [extra vllm flags...] -- (re)start the vLLM TPU server on :8000 with the sweep's fixed flags,
# then wait up to $BOOT_TIMEOUT seconds for /v1/models. Exit 0 when serving, 1 when the boot failed or timed
# out; the full container log is kept in $LOGS either way. Ported from ~/gemma4-dev/jev-tpu-v5e1/tpu/serve.sh.
#
# Every arm in a run gets the same --max-model-len (MAX_MODEL_LEN, default 4096) and --max-num-seqs 16, because
# attention cost follows --max-model-len on this chip and a throughput comparison is only fair at one setting.
# 4096 leaves room for the 2,048-token GSM8K limit on top of a prompt.
# IMAGE overrides the image. BOOT_TIMEOUT is seconds (default 2400). XLA_CACHE_DIR keeps the JAX compile cache
# on the host. MM_LIMIT sets --limit-mm-per-prompt and is empty by default, because vLLM rejects the flag for a
# text-only model; run_sweep.sh sets it for the multimodal checkpoints. VLLM_ENV, space-separated KEY=VALUE
# pairs, is passed into the container.
set -u
IMAGE=${IMAGE:?IMAGE must name the pinned image}
LOGS=${LOGS:-/opt/model-sweep-logs}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-2400}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MM_LIMIT=${MM_LIMIT:-}
MM_ARGS=(); [ -n "$MM_LIMIT" ] && MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT")
CACHE_ARGS=()
if [ -n "${XLA_CACHE_DIR:-}" ]; then
  mkdir -p "$XLA_CACHE_DIR"
  CACHE_ARGS=(-e VLLM_XLA_CACHE_PATH=/xla-cache -v "$XLA_CACHE_DIR":/xla-cache)
fi
ENV_ARGS=(); for kv in ${VLLM_ENV:-}; do ENV_ARGS+=(-e "$kv"); done
MODEL="$1"; shift
TAG=$(echo "$MODEL" | tr '/' '_')
mkdir -p "$LOGS"
docker rm -f vllm >/dev/null 2>&1
docker run -d --name vllm --privileged --net=host --shm-size 10gb -v /dev/shm:/dev/shm \
  -e HF_TOKEN="$(cat /root/hf_token)" -e HF_HOME=/dev/shm/hf "${CACHE_ARGS[@]}" "${ENV_ARGS[@]}" \
  "$IMAGE" \
  vllm serve "$MODEL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 16 \
    --generation-config vllm "${MM_ARGS[@]}" --enable-prefix-caching "$@" >/dev/null
START=$(date +%s)
while :; do
  if curl -sf localhost:8000/v1/models | grep -q '"id"'; then
    echo "READY $MODEL after $(( $(date +%s) - START ))s"; docker logs vllm > "$LOGS/$TAG.boot.log" 2>&1; exit 0
  fi
  if [ -z "$(docker ps -q -f name=vllm)" ] || [ $(( $(date +%s) - START )) -gt "$BOOT_TIMEOUT" ]; then
    docker logs vllm > "$LOGS/$TAG.boot.log" 2>&1
    # Quote the last raised exception (skipping vLLM's generic wrapper), not every line that mentions "error".
    if [ -z "$(docker ps -q -f name=vllm)" ]; then WHY="exited"; else WHY="timed out after ${BOOT_TIMEOUT}s"; fi
    echo "FAILED $MODEL ($WHY) after $(( $(date +%s) - START ))s: $(grep -av tpu_info "$LOGS/$TAG.boot.log" | grep -aoE '[A-Za-z_.]*(Error|Exception): .*|RESOURCE_EXHAUSTED.*' | grep -av 'See root cause above' | tail -1 | cut -c1-400)"
    docker rm -f vllm >/dev/null 2>&1; exit 1
  fi
  sleep 15
done
