#!/usr/bin/env bash
set -euo pipefail
REMOTE_DIR=${1:-$HOME/AELP/AELP2/outputs/demo_ads}
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg || true
cd "$REMOTE_DIR"
ffmpeg -y \
  -loop 1 -t 2 -i 01_hook.png \
  -loop 1 -t 3 -i 02_proof.png \
  -loop 1 -t 4 -i 03_claim.png \
  -loop 1 -t 3 -i 04_cta.png \
  -f lavfi -t 12 -i anoisesrc=color=pink:amplitude=0.003:seed=42 \
  -filter_complex "[0:v]scale=1080:1920,format=yuv420p[v0];[1:v]scale=1080:1920,format=yuv420p[v1];[2:v]scale=1080:1920,format=yuv420p[v2];[3:v]scale=1080:1920,format=yuv420p[v3];[v0][v1][v2][v3]concat=n=4:v=1:a=0,format=yuv420p[v]" \
  -map "[v]" -map 4:a -r 30 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 96k -shortest demo_identity.mp4
echo "wrote: $REMOTE_DIR/demo_identity.mp4"

