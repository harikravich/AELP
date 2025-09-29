#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

echo "[full-start] Starting Next.js API on :3000"
bash AELP2/scripts/aelp2ctl.sh stop || true
PORT=3000 bash AELP2/scripts/aelp2ctl.sh start

echo "[full-start] Starting Vite frontend on :8080 (proxy /api → :3000)"
pushd AELP2/external/growth-compass-77 >/dev/null
npm_config_production=false npm ci >/dev/null 2>&1 || npm_config_production=false npm install
nohup npm run dev >/tmp/aelp2_vite_8080.log 2>&1 &
popd >/dev/null

echo "[full-start] Ready: http://localhost:8080 (frontend) → http://localhost:3000/api"
echo "[logs] tail -f /tmp/aelp2_dashboard.log /tmp/aelp2_vite_8080.log"

