#!/usr/bin/env bash
set -euo pipefail

# Background GPU runner for creative CTR pipeline (uses CLIP/YOLO if CUDA is available).
# Usage:
#   ENABLE_CLIP=1 YOLO_DEVICE=cuda INSTALL_GPU_DEPS=1 \
#   scripts/bg_run_gpu_creative_ctr.sh

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env || true
fi

mkdir -p logs .runs artifacts

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/gpu_creative_ctr_${TS}.log"
PIDF=".runs/gpu_creative_ctr_${TS}.pid"

echo "[bg-gpu] Launching GPU creative CTR pipeline..."

set +e
env PYTHONPATH="$ROOT" \
  nohup bash -lc "scripts/gpu_creative_ctr.sh" \
  >"$LOG" 2>&1 &
RC=$!
set -e

echo $RC > "$PIDF"
echo "[bg-gpu] PID $(cat "$PIDF") | logfile: $LOG | pidfile: $PIDF"
echo "[bg-gpu] Tail logs: tail -f $LOG"

