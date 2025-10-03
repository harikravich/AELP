#!/bin/bash
set -euxo pipefail

# Cloud-init style startup for AELP on Debian 12

apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3.10 python3.10-venv python3-pip \
  build-essential git \
  libopenblas-dev liblapack-dev libatlas-base-dev \
  libffi-dev libssl-dev libsqlite3-dev libhdf5-dev \
  pkg-config

mkdir -p /opt/aelp/venvs /opt/aelp/work /var/log/aelp

# Create venvs
python3.10 -m venv /opt/aelp/venvs/aelp-heavy
python3.10 -m venv /opt/aelp/venvs/aelp-light

source /opt/aelp/venvs/aelp-light/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install reportlab pillow matplotlib requests
deactivate

source /opt/aelp/venvs/aelp-heavy/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Core scientific stack (pinned to avoid TF/Numba conflicts)
pip install numpy==1.24.4 numba==0.57.1 scipy==1.10.1 pandas==2.1.4 \
            scikit-learn==1.3.2 matplotlib==3.8.4 gymnasium==0.29.1
# TF/TFP for RecSim-NG
pip install tensorflow-cpu==2.18.0 tensorflow-probability==0.24.0
# RecSim-NG and extras
pip install recsim-ng==0.2.0 lifelines==0.28.0 SQLAlchemy==2.0.36 redis==5.0.7
# GCP clients
pip install google-cloud-bigquery google-cloud-storage google-cloud-pubsub
deactivate

echo "Startup bootstrap complete" | tee -a /var/log/aelp/startup.log

