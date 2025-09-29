#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_DIR="$ROOT_DIR/AELP2/config/skeletons"
mkdir -p "$CONFIG_DIR"

NAME=${1:-"Aura Meta – Sandbox Skeleton"}
OBJ=${2:-"lead_generation"}
BUDGET=${3:-50}
UTM_JSON=${4:-'{"utm_source":"meta","utm_medium":"cpc","utm_campaign":"skeleton"}'}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$CONFIG_DIR/meta_${STAMP}.json"

cat > "$OUT" << EOF
{
  "platform": "meta",
  "campaign_name": "$NAME",
  "objective": "$OBJ",
  "daily_budget": $BUDGET,
  "utm": $UTM_JSON,
  "status": "paused",
  "notes": "skeleton only, no spend"
}
EOF

echo "[META] Created skeleton payload: $OUT"

# Log to BigQuery
GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform} \
BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training} \
python3 -m AELP2.pipelines.platform_skeleton_log \
  --platform meta \
  --campaign_name "$NAME" \
  --objective "$OBJ" \
  --daily_budget "$BUDGET" \
  --notes "skeleton only" \
  --utm "$UTM_JSON"

echo "[META] Skeleton logged to BigQuery (shadow)."

