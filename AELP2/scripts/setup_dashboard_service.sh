#!/usr/bin/env bash
set -euo pipefail

# Creates a systemd service to run the dashboard on port 3001

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
APP_DIR="$ROOT_DIR/AELP2/apps/dashboard"

if ! command -v sudo >/dev/null 2>&1; then echo "sudo required" >&2; exit 1; fi

SERVICE_FILE=/etc/systemd/system/aelp2-dashboard.service

cat <<CONF | sudo tee $SERVICE_FILE >/dev/null
[Unit]
Description=AELP2 Dashboard (Next.js)
After=network.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
Environment=GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
Environment=BIGQUERY_TRAINING_DATASET=${BIGQUERY_TRAINING_DATASET:-gaelp_training}
Environment=BIGQUERY_USERS_DATASET=${BIGQUERY_USERS_DATASET:-gaelp_users}
Environment=GA4_PROPERTY_ID=${GA4_PROPERTY_ID:-}
Environment=HOST=0.0.0.0
Environment=PORT=3001
ExecStart=/usr/bin/npm start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
CONF

sudo systemctl daemon-reload
sudo systemctl enable aelp2-dashboard
sudo systemctl restart aelp2-dashboard
echo "Dashboard service running on 127.0.0.1:3001 (proxied via Nginx on :80)"

