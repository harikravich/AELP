#!/usr/bin/env bash
set -euo pipefail

# Background runner for the creative CTR pipeline.
# Usage:
#   UNIFIED=artifacts/synth/unified.parquet \
#   scripts/bg_run_creative_ctr.sh
#
# Optional envs:
#   ASSETS_DIR, CRE_FEATS, JOINED, CTR_MODEL, CTR_PREDS, NOVEL_JOINED, NOVEL_PREDS,
#   PRIORS, TS_OUT, TS_NOVEL_OUT, SLATE_OUT

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# Source .env if present (for any tokens, etc.)
if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env || true
fi

mkdir -p logs .runs artifacts

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/creative_ctr_${TS}.log"
PIDF=".runs/creative_ctr_${TS}.pid"

echo "[bg] Launching creative CTR pipeline..."

set +e
env PYTHONPATH="$ROOT" \
  nohup bash -lc "scripts/one_command_creative_ctr.sh" \
  >"$LOG" 2>&1 &
RC=$!
set -e

echo $RC > "$PIDF"
echo "[bg] PID $(cat "$PIDF") | logfile: $LOG | pidfile: $PIDF"
echo "[bg] Tail logs: tail -f $LOG"

