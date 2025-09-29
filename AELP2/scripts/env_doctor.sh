#!/usr/bin/env bash
set -euo pipefail

# Env Doctor: prints status of required env vars, GA4 service account, BigQuery access,
# and Cloud Run service account (if available).

PROJECT="${GOOGLE_CLOUD_PROJECT:-}"
DATASET="${BIGQUERY_TRAINING_DATASET:-}"
USERS_DS="${BIGQUERY_USERS_DATASET:-}"
GA4_PROP="${GA4_PROPERTY_ID:-}"
SERVICE="${1:-aelp2-dashboard}"
REGION="${REGION:-us-central1}"

echo "[doctor] Core env"
printf '  %-30s %s\n' GOOGLE_CLOUD_PROJECT "${PROJECT:-MISSING}" \
                    BIGQUERY_TRAINING_DATASET "${DATASET:-MISSING}" \
                    BIGQUERY_USERS_DATASET "${USERS_DS:-MISSING}" \
                    GA4_PROPERTY_ID "${GA4_PROP:-(optional)}" \
                    OPENAI_API_KEY "${OPENAI_API_KEY:+set}" \
                    GOOGLE_ADS_CUSTOMER_ID "${GOOGLE_ADS_CUSTOMER_ID:-(unset)}" \
                    GOOGLE_ADS_LOGIN_CUSTOMER_ID "${GOOGLE_ADS_LOGIN_CUSTOMER_ID:-(unset)}"

# GA4 service account detection (local dev)
SA_PATH="$HOME/.config/gaelp/ga4-service-account.json"
if [[ -f "$SA_PATH" ]]; then
  echo "[doctor] GA4 SA JSON: $SA_PATH (exists)"
else
  echo "[doctor] GA4 SA JSON: not found at $SA_PATH (ok if using Cloud Run SA)"
fi

if command -v gcloud >/dev/null 2>&1; then
  echo "[doctor] gcloud project: $(gcloud config get-value project 2>/dev/null || true)"
  # Cloud Run service account
  SA=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(spec.template.spec.serviceAccountName)' 2>/dev/null || true)
  if [[ -n "$SA" ]]; then
    echo "[doctor] Cloud Run SA: $SA"
  else
    echo "[doctor] Cloud Run SA: (service not found or no access)"
  fi
else
  echo "[doctor] gcloud not installed"
fi

if command -v bq >/dev/null 2>&1 && [[ -n "$PROJECT" && -n "$DATASET" ]]; then
  echo "[doctor] BigQuery dataset check: $PROJECT.$DATASET"
  (bq --project_id "$PROJECT" ls -d "$DATASET" >/dev/null 2>&1 && echo "  ✓ dataset exists") || echo "  ✗ dataset not found"
  echo "[doctor] Sample table presence (ads_kpi_daily)"
  (bq --project_id "$PROJECT" ls "$DATASET" | grep -q '^\s*ads_kpi_daily\b' && echo "  ✓ ads_kpi_daily exists" ) || echo "  ⚠ ads_kpi_daily not found (use /api/control/kpi-lock)"
else
  echo "[doctor] bq not available or env incomplete; skipping BQ checks"
fi

echo "[doctor] Done"

