#!/usr/bin/env bash
set -euo pipefail

# Build a 12s 9:16 MP4 on the VM from the storyboard PNGs with a soft pink-noise bed.
# Usage (local):
#   Z=us-central1-a VM=aelp-sim-rl-1 bash AELP2/tools/demo_make_ad_vm.sh

Z=${Z:-us-central1-a}
VM=${VM:-aelp-sim-rl-1}
REMOTE_DIR=~/AELP/AELP2/outputs/demo_ads

echo "[local] Ensuring storyboard exists"
python3 AELP2/tools/demo_storyboard.py >/dev/null || true

echo "[vm] Ensuring remote dir: $REMOTE_DIR"
gcloud compute ssh "$VM" --zone "$Z" --command "bash -lc 'mkdir -p $REMOTE_DIR'"

echo "[local] Copying frames to VM: $VM ($Z)"
gcloud compute scp AELP2/outputs/demo_ads/0{1..4}_*.png "$VM":$REMOTE_DIR --zone "$Z"

echo "[vm] Assembling MP4 via ffmpeg"
gcloud compute ssh "$VM" --zone "$Z" --command "bash -lc 'cat > ~/AELP/AELP2/tools/_demo_ffmpeg.sh <<\"EOS\"\nset -e\nsudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg || true\ncd $REMOTE_DIR\n# Inputs: 2s + 3s + 4s + 3s = 12s\nffmpeg -y \\n  -loop 1 -t 2 -i 01_hook.png \\n  -loop 1 -t 3 -i 02_proof.png \\n  -loop 1 -t 4 -i 03_claim.png \\n  -loop 1 -t 3 -i 04_cta.png \\n  -f lavfi -t 12 -i anoisesrc=color=pink:amplitude=0.003:seed=42 \\n  -filter_complex \"[0:v]scale=1080:1920,format=yuv420p[v0];[1:v]scale=1080:1920,format=yuv420p[v1];[2:v]scale=1080:1920,format=yuv420p[v2];[3:v]scale=1080:1920,format=yuv420p[v3];[v0][v1][v2][v3]concat=n=4:v=1:a=0,format=yuv420p[v]\" \\n  -map \"[v]\" -map 4:a -r 30 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 96k -shortest demo_identity.mp4\necho \"wrote: $REMOTE_DIR/demo_identity.mp4\"\nEOS\nchmod +x ~/AELP/AELP2/tools/_demo_ffmpeg.sh && bash ~/AELP/AELP2/tools/_demo_ffmpeg.sh'"

echo "[done] Preview on VM: $REMOTE_DIR/demo_identity.mp4"

