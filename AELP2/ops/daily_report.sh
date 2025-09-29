#!/usr/bin/env bash
set -euo pipefail
PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
DATASET="${BIGQUERY_TRAINING_DATASET:-}"
if [[ -z "$PROJECT" || -z "$DATASET" ]]; then echo "Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET"; exit 2; fi
echo "[Daily] Project=$PROJECT Dataset=$DATASET"
echo "[Freshness]";
python3 - <<PY
from google.cloud import bigquery
import os
PROJECT=os.environ['GOOGLE_CLOUD_PROJECT']; DATASET=os.environ['BIGQUERY_TRAINING_DATASET']
bq=bigquery.Client(project=PROJECT)
for t,f in [('ads_campaign_performance','date'),('ga4_daily','date'),('ab_metrics_daily','date'),('mmm_curves','timestamp')]:
  try:
    for r in bq.query(f"SELECT MAX({f}) AS d FROM `{PROJECT}.{DATASET}.{t}`").result():
      print(f"  {t}: {r['d']}")
  except Exception as e:
    print(f"  {t}: n/a ({e})")
PY
echo "[Jobs] (last run timestamps not tracked in this stub)"
