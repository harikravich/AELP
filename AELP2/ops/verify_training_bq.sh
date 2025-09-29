#!/usr/bin/env bash
set -euo pipefail

# Verify BigQuery rows after a training run and check canary readiness.
# Reads project/dataset from env or .env.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

# Load .env if present (does not override existing vars)
if [[ -f .env ]]; then
  while IFS='=' read -r key val; do
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < .env
fi

: "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT}"
: "${BIGQUERY_TRAINING_DATASET:?Set BIGQUERY_TRAINING_DATASET}"

PROJECT="$GOOGLE_CLOUD_PROJECT"
DS="$BIGQUERY_TRAINING_DATASET"

echo "Project: $PROJECT"
echo "Dataset: $DS"

echo "\n[1/5] training_episodes count + last timestamp"
bq query --use_legacy_sql=false --format=pretty "SELECT COUNT(*) AS episodes, MAX(TIMESTAMP(timestamp)) AS last_ts FROM \`$PROJECT.$DS.training_episodes\`" || true

echo "\n[2/5] bidding_events count + last timestamp"
bq query --use_legacy_sql=false --format=pretty "SELECT COUNT(*) AS bids, MAX(TIMESTAMP(timestamp)) AS last_ts FROM \`$PROJECT.$DS.bidding_events\`" || true

echo "\n[3/5] safety_events count + last timestamp"
bq query --use_legacy_sql=false --format=pretty "SELECT COUNT(*) AS safety, MAX(TIMESTAMP(timestamp)) AS last_ts FROM \`$PROJECT.$DS.safety_events\`" || true

echo "\n[4/5] latest training_runs record"
bq query --use_legacy_sql=false --format=pretty "SELECT run_id, status, episodes_completed, TIMESTAMP(end_time) AS end_time FROM \`$PROJECT.$DS.training_runs\` ORDER BY end_time DESC NULLS LAST LIMIT 1" || true

echo "\n[5/5] Canary readiness (fidelity + 14d ROAS)"
export AELP2_FIDELITY_WINDOW_DAYS="${AELP2_FIDELITY_WINDOW_DAYS:-14}"
python3 -m AELP2.scripts.check_canary_readiness || true

echo "\nIf counts are zero:"
echo "- Ensure the VM service account has BigQuery Data Editor on project $PROJECT"
echo "- Then rerun a 1-episode seed and re-run this script"

echo "\nFetch VM service account and show grant command (for reference):"
SA=$(curl -H "Metadata-Flavor: Google" -s http://metadata/computeMetadata/v1/instance/service-accounts/default/email || true)
if [[ -n "$SA" ]]; then
  echo "Service Account: $SA"
  echo "Grant (one-time):"
  echo "  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$SA --role=roles/bigquery.dataEditor"
fi

echo "\nDone."

