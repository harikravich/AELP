#!/usr/bin/env bash
set -euo pipefail

# Background trainer for Avazu_x1 with logging and PID tracking.
# Usage:
#   HUGGINGFACE_TOKEN=... scripts/bg_train_avazu.sh \
#     --train-rows 5000000 --test-rows 2000000
#
# Optional: put HUGGINGFACE_TOKEN in .env (gitignored). This script will source it if present.

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env || true
fi

mkdir -p logs .runs artifacts/models artifacts/avazu

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/avazu_train_${TS}.log"
PIDF=".runs/avazu_train_${TS}.pid"

echo "[bg] Launching Avazu_x1 training..."
set +e
env PYTHONPATH="$ROOT" HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
  nohup python3 pipelines/third_party/avazu_train_eval.py "$@" \
  >"$LOG" 2>&1 &
RC=$!
set -e
echo $RC > "$PIDF"
echo "[bg] PID $(cat "$PIDF") | logfile: $LOG | pidfile: $PIDF"
echo "[bg] Tail logs: tail -f $LOG"
