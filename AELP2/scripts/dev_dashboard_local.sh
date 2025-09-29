#!/usr/bin/env bash
set -euo pipefail

# Start the dashboard locally on the VM for preview without Cloud Build.
# - Installs deps, builds, and serves on port 3001
# - Uses env from .env if present, otherwise reads from the shell

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR/AELP2/apps/dashboard"

# Non-destructive env load from repo .env
if [[ -f "$ROOT_DIR/.env" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    line="${line#export }"
    key="${line%%=*}"; val="${line#*=}"
    [[ -z "$key" ]] && continue
    if [[ -z "${!key:-}" ]]; then export "$key=$val"; fi
  done < "$ROOT_DIR/.env"
fi

export GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
export BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
export BIGQUERY_USERS_DATASET=${BIGQUERY_USERS_DATASET:-gaelp_users}
export GA4_PROPERTY_ID=${GA4_PROPERTY_ID:-}

echo "Project=$GOOGLE_CLOUD_PROJECT  Dataset=$BIGQUERY_TRAINING_DATASET  Users=$BIGQUERY_USERS_DATASET  GA4=$GA4_PROPERTY_ID"

echo "Installing deps…"
# Force install devDependencies even if NODE_ENV=production is exported
npm_config_production=false npm install --no-audit --no-fund

echo "Building…"
NODE_ENV=production npm run build

echo "Starting on http://localhost:3001 … (use SSH tunnel: -L 3001:localhost:3001)"
HOST=0.0.0.0 PORT=3001 npm start
