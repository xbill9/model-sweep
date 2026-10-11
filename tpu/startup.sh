#!/bin/bash
# Startup script for an unattended sweep on a v5e-1 queued resource (ported from
# ~/gemma4-dev/jev-tpu-v5e1/tpu/startup_quant.sh): the code bundle named by metadata sweep-code into
# /opt/model-sweep, Docker, the HF token from Secret Manager, then run_sweep.sh with metadata sweep-run.
exec > >(tee -a /var/log/sweep-boot.log /dev/console) 2>&1
set -u
md() { curl -s -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/$1"; }
PROJECT=$(md project/project-id)
mkdir -p /opt/model-sweep
if ! gcloud storage cp -q "$(md instance/attributes/sweep-gcs)/inputs/$(md instance/attributes/sweep-code)" /tmp/code.tgz \
   || ! tar xzf /tmp/code.tgz -C /opt/model-sweep; then
  echo "code bundle $(md instance/attributes/sweep-code) missing or unreadable; deleting this node"
  QR=$(md instance/attributes/sweep-qr); ZONE=$(md instance/zone | awk -F/ '{print $NF}')
  [ -n "$QR" ] && gcloud alpha compute tpus queued-resources delete "$QR" --zone "$ZONE" --force --quiet --async
  exit 1
fi
command -v docker >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io >/dev/null; }
for _ in $(seq 1 60); do
  AT=$(md instance/service-accounts/default/token | python3 -c 'import json,sys;print(json.load(sys.stdin)["access_token"])')
  HF_TOKEN=$(curl -s -H "Authorization: Bearer $AT" "https://secretmanager.googleapis.com/v1/projects/$PROJECT/secrets/hf-token/versions/latest:access" | python3 -c 'import json,sys,base64;print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode())' 2>/dev/null)
  [ -n "$HF_TOKEN" ] && break; sleep 30
done
[ -n "$HF_TOKEN" ] || { echo "no hf-token"; exit 1; }
install -m 600 /dev/null /root/hf_token && printf '%s' "$HF_TOKEN" > /root/hf_token
bash /opt/model-sweep/tpu/run_sweep.sh "$(md instance/attributes/sweep-run)"
