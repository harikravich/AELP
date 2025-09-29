#!/usr/bin/env bash
set -euo pipefail

APP_DIR="AELP2/apps/dashboard"
SERVICE_NAME="aelp2-dashboard"
REGION="${REGION:-us-central1}"

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
  echo "Set GOOGLE_CLOUD_PROJECT" >&2; exit 2
fi

# Optional: NextAuth / Google OAuth for SSO
# Required for protected access:
#   NEXTAUTH_URL, NEXTAUTH_SECRET, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
# Optional:
#   ALLOWED_EMAIL_DOMAIN (e.g., example.com)
# Optional: specify a service account to run as (must have BQ read perms): AELP2_DASHBOARD_SA

pushd "$APP_DIR" >/dev/null
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/$SERVICE_NAME:$(date +%s)
IMG=$(gcloud container images list-tags gcr.io/$GOOGLE_CLOUD_PROJECT/$SERVICE_NAME --format='value(TAGS)' --limit=1)

DEPLOY_ARGS=(
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/$SERVICE_NAME:$IMG
  --platform managed
  --region "$REGION"
  --allow-unauthenticated
  --set-env-vars GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT
  --set-env-vars BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
)

if [[ -n "${NEXTAUTH_URL:-}" ]]; then DEPLOY_ARGS+=(--set-env-vars NEXTAUTH_URL="$NEXTAUTH_URL"); fi
if [[ -n "${NEXTAUTH_SECRET:-}" ]]; then DEPLOY_ARGS+=(--set-env-vars NEXTAUTH_SECRET="$NEXTAUTH_SECRET"); fi
if [[ -n "${GOOGLE_CLIENT_ID:-}" ]]; then DEPLOY_ARGS+=(--set-env-vars GOOGLE_CLIENT_ID="$GOOGLE_CLIENT_ID"); fi
if [[ -n "${GOOGLE_CLIENT_SECRET:-}" ]]; then DEPLOY_ARGS+=(--set-env-vars GOOGLE_CLIENT_SECRET="$GOOGLE_CLIENT_SECRET"); fi
if [[ -n "${ALLOWED_EMAIL_DOMAIN:-}" ]]; then DEPLOY_ARGS+=(--set-env-vars ALLOWED_EMAIL_DOMAIN="$ALLOWED_EMAIL_DOMAIN"); fi
if [[ -n "${AELP2_DASHBOARD_SA:-}" ]]; then DEPLOY_ARGS+=(--service-account "$AELP2_DASHBOARD_SA"); fi

gcloud run deploy "$SERVICE_NAME" "${DEPLOY_ARGS[@]}"
popd >/dev/null
echo "Deployed $SERVICE_NAME in $REGION"
