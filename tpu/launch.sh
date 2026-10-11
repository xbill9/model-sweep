#!/bin/bash
# launch.sh <arms-file> [--go] -- bundle this tree, upload it, and create the flex-start v5e-1 queued resource
# that runs the sweep and deletes itself. Without --go it prints what it would do and creates nothing.
#
#   tpu/launch.sh arms-v5e1-tier1.txt          # dry run
#   tpu/launch.sh arms-v5e1-tier1.txt --go     # upload and create
#
# Values come from tpu/sweep.env (a real environment variable wins). The bundle holds the committed tree only
# (git archive HEAD) and is named by its content hash, so a run always records exactly which code it ran.
# Results land in $GCS_ROOT/<run>/; follow the boot with
#   gcloud alpha compute tpus tpu-vm get-serial-port-output <qr>-node --zone <zone>   (or the console)
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ARMS=${1:?usage: launch.sh <arms-file in tpu/> [--go]}
GO=${2:-}
[ -f "$HERE/tpu/$ARMS" ] || { echo "no such arms file: tpu/$ARMS" >&2; exit 1; }
while IFS='=' read -r k v; do
  [[ "$k" =~ ^[A-Z_]+$ ]] && [ -z "${!k:-}" ] && export "$k=$v"
done < <(grep -E '^[A-Z_]+=' "$HERE/tpu/sweep.env")

cd "$HERE"
if [ -n "$(git status --porcelain -- . ':!LINEUP.md')" ]; then
  echo "the tree has uncommitted changes; the bundle is built from HEAD, so commit first" >&2; exit 1
fi
python3 -m unittest discover -s tests >/dev/null 2>&1 || { echo "unit tests fail; run python3 -m unittest discover -s tests -v" >&2; exit 1; }

COMMIT=$(git rev-parse --short=9 HEAD)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
git archive --format=tar.gz -o "$TMP/bundle.tgz" HEAD
CODE=sweep-$COMMIT-$(sha256sum "$TMP/bundle.tgz" | cut -c1-8).tgz
STEM=${ARMS%.txt}; STEM=${STEM#arms-}
RUN=$(date -u +%Y-%m-%d)-$STEM
QR=ms-$RUN-$(date -u +%H%M)

META="sweep-code=$CODE,sweep-run=$RUN,sweep-qr=$QR,sweep-gcs=$GCS_ROOT,sweep-image=$IMAGE,sweep-arms=$ARMS"
CMD=(gcloud alpha compute tpus queued-resources create "$QR" --project "$GOOGLE_CLOUD_PROJECT"
  --zone "$GOOGLE_CLOUD_ZONE" --accelerator-type "$ACCELERATOR_TYPE" --runtime-version "$RUNTIME_VERSION"
  --node-id "$QR-node" --provisioning-model flex-start --max-run-duration "$MAX_RUN_DURATION"
  --valid-until-duration "$VALID_UNTIL_DURATION" --metadata "$META"
  --metadata-from-file startup-script=tpu/startup.sh)

echo "arms:    tpu/$ARMS ($(grep -cvE '^\s*(#|$)' "tpu/$ARMS") arms)"
echo "bundle:  $GCS_ROOT/inputs/$CODE"
echo "results: $GCS_ROOT/$RUN/"
printf 'create: '; printf '%q ' "${CMD[@]}"; echo
if [ "$GO" != --go ]; then echo "dry run: nothing uploaded or created (pass --go)"; exit 0; fi

gcloud storage cp -q "$TMP/bundle.tgz" "$GCS_ROOT/inputs/$CODE"
"${CMD[@]}"
echo "created $QR; it deletes itself when the last arm finishes, or at $MAX_RUN_DURATION"
