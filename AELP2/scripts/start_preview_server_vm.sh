#!/usr/bin/env bash
set -euo pipefail
# Start a previews HTTP server on a GCE VM at 127.0.0.1:8080 serving AELP2/outputs/finals
# Usage: ./AELP2/scripts/start_preview_server_vm.sh <instance> <zone>
VM=${1:-aelp-sim-rl-1}
ZONE=${2:-us-central1-a}
gcloud compute ssh "$VM" --zone "$ZONE" --command "bash -lc 'nohup python3 ~/AELP/AELP2/tools/serve_previews.py >/tmp/preview_server.log 2>&1 & echo started'"
echo "Now tunnel: gcloud compute ssh $VM --zone $ZONE -- -N -L 8080:127.0.0.1:8080"
