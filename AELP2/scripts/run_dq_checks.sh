#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then set -a; source .env; set +a; fi

echo "Running data quality checks..."
python3 -m AELP2.pipelines.check_data_quality || true

