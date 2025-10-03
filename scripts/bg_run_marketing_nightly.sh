#!/usr/bin/env bash
set -euo pipefail

# Nightly loop for Marketing API creative CTR pipeline.
# Runs once per 24 hours using Marketing artifacts.
# Usage:
#   scripts/bg_run_marketing_nightly.sh &

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if [[ -f .env ]]; then
  # shellcheck disable=SC1091
  source .env || true
fi

mkdir -p logs .runs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/nightly_marketing_${TS}.log"
PIDF=".runs/nightly_marketing_${TS}.pid"

(
  while true; do
    echo "[nightly] $(date -Is) start" | tee -a "$LOG"
    UNIFIED=artifacts/marketing/unified_ctr.parquet \
    ASSETS_DIR=assets/meta_creatives \
    CRE_FEATS=artifacts/creative/meta_creative_features.parquet \
    JOINED=artifacts/features/marketing_ctr_joined.parquet \
    CTR_MODEL=artifacts/models/ctr_creative_marketing.joblib \
    CTR_PREDS=artifacts/predictions/ctr_scores.parquet \
    NOVEL_JOINED=artifacts/features/meta_catalog.parquet \
    NOVEL_PREDS=artifacts/predictions/novel_ctr_scores.parquet \
    PRIORS=artifacts/priors/priors.json \
    TS_OUT=artifacts/priors/ts_strategies_ctr.json \
    TS_NOVEL_OUT=artifacts/priors/ts_strategies_novel.json \
    SLATE_OUT=artifacts/slates/ad_slate.json \
      bash scripts/one_command_creative_ctr.sh >> "$LOG" 2>&1 || true
    echo "[nightly] $(date -Is) done" | tee -a "$LOG"
    sleep 86400
  done
) &

echo $! > "$PIDF"
echo "[nightly] PID $(cat "$PIDF") | logfile: $LOG"

