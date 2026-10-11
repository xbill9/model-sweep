#!/bin/bash
# run_sweep.sh <run-prefix> -- serve each arm in turn on the v5e-1 TPU VM, measure it, upload, then delete the
# queued resource. Ported from ~/gemma4-dev/jev-tpu-v5e1/tpu/run_quant.sh (b584cd5), keeping its order of work
# and its self-delete, and dropping what only the Gemma repack study needed: tpu_inference patches, the label
# read and Nimble suite, compile-cache seeding, GCS checkpoints and swap.
#
# Metadata (set by launch.sh): sweep-gcs (GCS root), sweep-image (pinned image), sweep-arms (an arms file in
# tpu/, e.g. arms-v5e1-tier1.txt), sweep-qr (the queued resource to delete at the end). Optional:
# sweep-load-conc (concurrencies for load and long, default "1 4 16"), sweep-long-input (prompt lengths for
# long, default "1024 3072"), sweep-gen-max-tokens (GSM8K answer limit, default 2048), sweep-boot-timeout
# (seconds per arm, default 2400), sweep-env (KEY=VALUE pairs for every vLLM container).
#
# Arms file: one arm per line, `<MODEL_NAME>=<tag>=<modes>[=<flags>]`, # starts a comment. <tag> is the rig's
# slot 4 (NAMING.md). <modes> is a +-joined list of
#   load  fixed work: 256 output tokens per request, ignore_eos, at each sweep-load-conc concurrency
#   long  the same with long prompts that never hit the prefix cache, at each sweep-long-input length
#   gen   GSM8K (1,319 records), plus BFCL simple (400) when the arm names a tool parser; scored in code
# <flags> is comma-separated:
#   parser=<name>  --enable-auto-tool-choice --tool-call-parser <name>; without it gen skips BFCL
#   nothink        send chat_template_kwargs {"enable_thinking": false} on every request
#   mm=<a+b>       --limit-mm-per-prompt with each named modality at 0 (multimodal checkpoints, text-only arm)
#   gmu=<x>        --gpu-memory-utilization for this arm only
set -u
PREFIX="$1"
W=${SWEEP_DIR:-/opt/model-sweep}; L=${SWEEP_LOGS:-/opt/model-sweep-logs}; mkdir -p $L $W/results
export LOGS=$L
LOG=$L/run.log
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
attr() { local v; v=$(md attributes/$1); [ -n "$v" ] && [ "${v:0:1}" != "<" ] && printf '%s' "$v"; }
ZONE=$(md zone | awk -F/ '{print $NF}')
log() { echo "[sweep $(date -u +%FT%TZ)] $*" | tee -a $LOG > /dev/console; }
GCS=$(attr sweep-gcs) || { log "no sweep-gcs metadata"; exit 1; }
DST="$GCS/$PREFIX"
sync_up() { gcloud storage rsync -r -q $W/results "$DST/results" >/dev/null 2>&1; gcloud storage rsync -r -q $L "$DST/logs" >/dev/null 2>&1; }
finish() {
  log "DONE: $1"; sync_up
  local qr
  if ! qr=$(attr sweep-qr); then log "no sweep-qr metadata: leaving the node up, delete it by hand"; exit 0; fi
  gcloud alpha compute tpus queued-resources delete "$qr" --zone "$ZONE" --force --quiet --async >> $LOG 2>&1
  sync_up; exit 0
}

export IMAGE; IMAGE=$(attr sweep-image) || finish "no sweep-image metadata"
export BOOT_TIMEOUT; BOOT_TIMEOUT=$(attr sweep-boot-timeout) || BOOT_TIMEOUT=2400
export VLLM_ENV; VLLM_ENV=$(attr sweep-env) || VLLM_ENV=
export XLA_CACHE_DIR=/opt/xla-cache
CONCS=$(attr sweep-load-conc) || CONCS="1 4 16"
LONG_INPUT=$(attr sweep-long-input) || LONG_INPUT="1024 3072"
GEN_MAX=$(attr sweep-gen-max-tokens) || GEN_MAX=2048
ARMS_FILE=$(attr sweep-arms) || finish "no sweep-arms metadata"
[ -f "$W/tpu/$ARMS_FILE" ] || finish "arms file tpu/$ARMS_FILE is not in the bundle"

log "start $PREFIX on $ZONE, arms $ARMS_FILE"
for i in 1 2 3 4 5; do docker pull -q "$IMAGE" >> $LOG 2>&1 && break; log "image pull attempt $i failed"; sleep 30; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || finish "image pull failed"
log "image $(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
docker run --rm --entrypoint python3 "$IMAGE" -c "import vllm;print('vllm',vllm.__version__)" > $L/vllm-version.txt 2>&1
log "$(tail -1 $L/vllm-version.txt)"
sync_up

try_arm() {  # try_arm <model> <tag> <modes> [flags]
  local model=$1 tag=$2 mode=$3 flags=${4:-} parser='' kwargs='' mm='' f
  local extra=()
  has() { [[ "+$mode+" == *"+$1+"* ]]; }
  for f in ${flags//,/ }; do
    case "$f" in
      parser=*) parser=${f#parser=}; extra+=(--enable-auto-tool-choice --tool-call-parser "$parser") ;;
      nothink) kwargs='{"enable_thinking": false}' ;;
      mm=*) mm=$(python3 -c 'import json,sys;print(json.dumps({m: 0 for m in sys.argv[1].split("+")}))' "${f#mm=}") ;;
      gmu=*) extra+=(--gpu-memory-utilization "${f#gmu=}") ;;
      *) log "$tag: unknown arm flag $f, skipping the arm"; return ;;
    esac
  done
  export SWEEP_CHAT_TEMPLATE_KWARGS=$kwargs
  # The exact revision served, for the rig's README: the Hub's current commit when the arm starts.
  curl -s "https://huggingface.co/api/models/$model" | python3 -c \
    'import json,sys;d=json.load(sys.stdin);print(json.dumps({"model":d["id"],"sha":d.get("sha"),"license":(d.get("cardData") or {}).get("license")}))' \
    > $L/$tag.meta.json 2>/dev/null
  log "serving $model as $tag ($mode${flags:+, $flags})"
  ( while :; do echo "$(date -u +%T) $(free -m | awk '/^Mem:/{print "used",$3,"avail",$7}')"; sleep 30; done ) > $L/$tag.hostmem.txt 2>&1 &
  local memlog=$!
  if MM_LIMIT=$mm bash $W/tpu/serve.sh "$model" "${extra[@]}" >> $LOG 2>&1; then
    log "$(tail -1 $LOG)"
    if has load; then
      for c in $CONCS; do
        SWEEP_LOAD_CONCURRENCY=$c python3 $W/tpu/load_client.py load "$model" $L/$tag.load.c$c.json > /dev/null 2>> $LOG
        log "$tag load at concurrency $c: $(python3 -c "import json;d=json.load(open('$L/$tag.load.c$c.json'));print(d['output_tok_per_s'],'tok/s, range',d['output_tok_per_s_min'],'to',d['output_tok_per_s_max'])" 2>&1)"
      done
      sync_up
    fi
    if has long; then
      for n in $LONG_INPUT; do
        for c in $CONCS; do
          SWEEP_LOAD_INPUT_TOKENS=$n SWEEP_LOAD_CONCURRENCY=$c python3 $W/tpu/load_client.py loadlong "$model" $L/$tag.long$n.c$c.json > /dev/null 2>> $LOG
          log "$tag long $n at concurrency $c: $(python3 -c "import json;d=json.load(open('$L/$tag.long$n.c$c.json'));print(d['prompt_tokens_per_request'],'prompt tokens,',d['output_tok_per_s'],'out tok/s, ttft',d['ttft_median_s'],'s')" 2>&1)"
        done
      done
      sync_up
    fi
    if has gen; then
      local tasks=gsm8k; [ -n "$parser" ] && tasks="gsm8k bfcl_simple"
      for task in $tasks; do
        SWEEP_GEN_MAX_TOKENS=$GEN_MAX python3 $W/gen_eval.py run http://localhost:8000 "$model" $task \
          $W/results/$PREFIX-$tag-gen/$task.jsonl > $L/$tag.gen-$task.json 2>> $LOG
        log "$tag $task: $(cat $L/$tag.gen-$task.json)"
      done
      sync_up
    fi
  else
    log "$(grep -E '^(FAILED|READY)' $LOG | tail -1)"
    dmesg -T 2>/dev/null | grep -iE 'out of memory|oom-kill|killed process' > $L/$tag.oom.txt
    log "$tag host memory at the end: $(tail -1 $L/$tag.hostmem.txt); kernel OOM lines: $(wc -l < $L/$tag.oom.txt)"
  fi
  kill $memlog 2>/dev/null
  cp "$L/$(echo "$model" | tr '/' '_').boot.log" "$L/$tag.boot.log" 2>/dev/null
  docker rm -f vllm >/dev/null 2>&1; rm -rf /dev/shm/hf; sync_up
}

grep -vE '^\s*(#|$)' "$W/tpu/$ARMS_FILE" | while IFS='=' read -r model tag mode flags; do
  try_arm "$model" "$tag" "$mode" "$flags" < /dev/null
done
finish "all arms attempted"
