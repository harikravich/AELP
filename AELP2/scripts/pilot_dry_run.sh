#!/usr/bin/env bash
set -euo pipefail
BASE="${BASE_URL:-http://localhost:3000}"

COOKIE_JAR="${COOKIE_JAR:-/tmp/aelp2_pilot.cookies}"
rm -f "$COOKIE_JAR" 2>/dev/null || true

# Ensure server uses Sandbox dataset for write actions (via cookie)
curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "$BASE/api/dataset?mode=sandbox" >/dev/null || true

echo "[Pilot] Enqueue demo creative (PAUSED)"
curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "$BASE/api/control/creative/enqueue" \
  -H 'Content-Type: application/json' \
  -d '{"platform":"google_ads","type":"rsa","campaign_id":"demo","ad_group_id":"demo","payload":{"headlines":["See Online Balance"],"descriptions":["Help your teen thrive"],"final_url":"https://example.com/balance"}}' | tee /tmp/pilot_enqueue.json

RUN_ID=$(jq -r '.run_id' /tmp/pilot_enqueue.json 2>/dev/null || echo "")
if [[ -n "$RUN_ID" && "$RUN_ID" != "null" ]]; then
  echo "[Pilot] Approve publish (will log paused_created)"
  curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "$BASE/api/control/creative/publish" -F run_id="$RUN_ID" | cat
fi

echo "[Pilot] Publish LP A/B demo"
curl -s -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "$BASE/api/control/lp/publish?lp_a=/balance-v2&lp_b=/balance-insight-v1&traffic_split=0.5&primary_metric=cac" | cat

echo "[Pilot] Done. Review /approvals, /experiments, and /landing."
