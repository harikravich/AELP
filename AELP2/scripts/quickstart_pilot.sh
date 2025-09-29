#!/usr/bin/env bash
set -euo pipefail

echo "[Quickstart] AELP2 Pilot (Local)"

REQ=(GOOGLE_CLOUD_PROJECT BIGQUERY_TRAINING_DATASET BIGQUERY_USERS_DATASET)
for v in "${REQ[@]}"; do
  if [[ -z "${!v:-}" ]]; then echo "Missing env: $v" >&2; exit 2; fi
done

PORT=${PORT:-3000}
BASE_URL=${BASE_URL:-http://localhost:$PORT}

echo "[1/5] Installing Python deps (google-cloud-bigquery)"
python3 -m pip install --user -q google-cloud-bigquery google-auth google-api-core >/dev/null 2>&1 || true

echo "[2/5] Applying BigQuery schemas"
python3 AELP2/scripts/apply_schemas.py

echo "[3/5] Starting dashboard on $BASE_URL (background)"
pushd AELP2/apps/dashboard >/dev/null
# Ensure devDependencies are installed even if NODE_ENV=production is set in .env
npm_config_production=false npm ci >/dev/null 2>&1 || npm_config_production=false npm install
# Free the port if a stale Next process is running
if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[quickstart] Found process on port $PORT; stopping it"
  kill $(lsof -tiTCP:"$PORT" -sTCP:LISTEN) >/dev/null 2>&1 || true
  sleep 1
fi
# Source repo .env so Google Ads / GA4 creds are available to API routes
set +u
if [ -f "../../.env" ]; then
  echo "[quickstart] Loading ../../.env"
  set -a
  # shellcheck disable=SC1091
  . ../../.env
  set +a
fi
set -u
nohup env NODE_ENV=development NEXT_PUBLIC_BASE_URL="$BASE_URL" PILOT_MODE=1 PORT="$PORT" AELP2_REDACT_CAMPAIGN_NAMES=0 npm run dev >/tmp/aelp2_dashboard.log 2>&1 &
S=$!
popd >/dev/null

echo "[4/5] Waiting for dashboard to be ready..."
for i in {1..60}; do
  if curl -fsS "$BASE_URL" >/dev/null; then echo "Dashboard is up."; break; fi
  sleep 1
done

echo "[5/5] Running pilot dry-run"
env BASE_URL="$BASE_URL" bash AELP2/scripts/pilot_dry_run.sh || true

cat <<EOT

Quickstart done.
- Open $BASE_URL
- Visit: /ops/chat, /approvals, /experiments, /landing, /spend-planner, /channels, /audiences, /rl-insights, /backstage
- Logs: tail -f /tmp/aelp2_dashboard.log

To stop dashboard: kill $S
EOT
