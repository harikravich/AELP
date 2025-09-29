#!/usr/bin/env bash
set -euo pipefail

# Install a nightly cron to run AELP2/scripts/nightly_jobs.sh at 02:15 local time.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
JOB="bash $ROOT_DIR/AELP2/scripts/nightly_jobs.sh > /tmp/aelp2_nightly.log 2>&1"

TMP_CRON=$(mktemp)
crontab -l 2>/dev/null | grep -v 'AELP2/scripts/nightly_jobs.sh' > "$TMP_CRON" || true
echo "15 2 * * * $JOB" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

echo "Installed cron entry: 15 2 * * * $JOB"
