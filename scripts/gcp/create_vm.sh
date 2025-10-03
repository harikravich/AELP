#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PROJECT_ID=your-project \
#   REGION=us-central1 \
#   ZONE=us-central1-a \
#   SA_EMAIL=aelp-runner@your-project.iam.gserviceaccount.com \
#   VM_NAME=aelp-sim-1 \
#   ./scripts/gcp/create_vm.sh

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:?Set REGION}"
: "${ZONE:?Set ZONE}"
: "${SA_EMAIL:?Set SA_EMAIL}"
: "${VM_NAME:?Set VM_NAME}"

echo "Creating VM ${VM_NAME} in ${PROJECT_ID}/${ZONE} ..."

gcloud config set project "${PROJECT_ID}" >/dev/null

# Create a startup script file from our template
TMP_STARTUP=$(mktemp)
cp scripts/gcp/startup.sh "${TMP_STARTUP}"

# Create the VM with generous disk for TF/TFP/RecSim stacks
gcloud compute instances create "${VM_NAME}" \
  --zone="${ZONE}" \
  --machine-type="n2-standard-16" \
  --image-family="debian-12" --image-project="debian-cloud" \
  --boot-disk-size=300GB \
  --service-account="${SA_EMAIL}" \
  --scopes="https://www.googleapis.com/auth/cloud-platform" \
  --metadata-from-file startup-script="${TMP_STARTUP}"

echo "VM created. Use: gcloud compute ssh ${VM_NAME} --zone ${ZONE}"

