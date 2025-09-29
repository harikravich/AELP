#!/usr/bin/env bash
set -euo pipefail

# Simple data quality checks for core tables over the last N days.
# Usage: GOOGLE_CLOUD_PROJECT=... BIGQUERY_TRAINING_DATASET=... bash AELP2/scripts/dq_check.sh [DAYS]

DAYS=${1:-14}

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" || -z "${BIGQUERY_TRAINING_DATASET:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET" >&2; exit 2
fi

DS="$GOOGLE_CLOUD_PROJECT.$BIGQUERY_TRAINING_DATASET"

echo "[DQ] training_episodes recent counts"
bq query --use_legacy_sql=false --format=table "
SELECT DATE(timestamp) date, COUNT(*) rows, SUM(spend) spend, SUM(revenue) revenue, SUM(conversions) conv
FROM \`$DS.training_episodes\`
WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL $DAYS DAY) AND CURRENT_DATE()
GROUP BY date ORDER BY date DESC LIMIT 14" || true

echo "[DQ] ads_campaign_performance freshness and null checks"
bq query --use_legacy_sql=false --format=table "
SELECT COUNTIF(impressions IS NULL) null_impressions,
       COUNTIF(clicks IS NULL) null_clicks,
       COUNTIF(cost_micros IS NULL) null_cost,
       COUNTIF(conversions < 0 OR conversion_value < 0) negative_conversions
FROM \`$DS.ads_campaign_performance\`
WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL $DAYS DAY) AND CURRENT_DATE()" || true

echo "[DQ] ga4_aggregates freshness"
bq query --use_legacy_sql=false --format=table "
SELECT DATE(date) date, SUM(sessions) sessions, SUM(conversions) conversions
FROM \`$DS.ga4_aggregates\`
WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
GROUP BY date ORDER BY date DESC LIMIT 14" || true

echo "[DQ] ads_conversion_action_stats presence"
bq query --use_legacy_sql=false --format=table "
SELECT DATE(date) date, COUNT(*) rows
FROM \`$DS.ads_conversion_action_stats\`
WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND CURRENT_DATE()
GROUP BY date ORDER BY date DESC LIMIT 14" || true

echo "DQ checks completed."

