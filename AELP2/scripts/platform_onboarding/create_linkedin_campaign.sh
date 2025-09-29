#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONFIG_DIR="$ROOT_DIR/AELP2/config/skeletons"
mkdir -p "$CONFIG_DIR"

NAME=${1:-"Aura LinkedIn – Sandbox Skeleton"}
OBJ=${2:-"website_visits"}
BUDGET=${3:-50}
UTM_JSON=${4:-'{"utm_source":"linkedin","utm_medium":"cpc","utm_campaign":"sandbox_skeleton"}'}

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$CONFIG_DIR/linkedin_${STAMP}.json"

cat > "$OUT" << EOF
{
  "platform": "linkedin",
  "campaign_name": "$NAME",
  "objective": "$OBJ",
  "daily_budget": $BUDGET,
  "utm": $UTM_JSON,
  "status": "paused",
  "notes": "skeleton only, no spend"
}
EOF

echo "[LINKEDIN] Created skeleton payload: $OUT"

GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform} \
BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training} \
python3 -m AELP2.pipelines.platform_skeleton_log \
  --platform linkedin \
  --campaign_name "$NAME" \
  --objective "$OBJ" \
  --daily_budget "$BUDGET" \
  --notes "skeleton only" \
  --utm "$UTM_JSON"

echo "[LINKEDIN] Skeleton logged to BigQuery (shadow)."

