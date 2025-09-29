# RL-sim GCP Instance Inventory — 2025-09-28

Instance: `aelp-sim-rl-1`
- Project: `aura-thrive-platform`
- Zone: `us-central1-a`
- Machine: `n2-standard-16` (16 vCPU, 64 GB RAM) running Debian 12 (startup script provisions Python 3.10 venvs).
- Disk: 300 GB persistent disk (`aelp-sim-rl-1`).
- Service account: `556751870393-compute@developer.gserviceaccount.com` (cloud-platform scope).
- External IP: `34.59.203.14`, Internal: `10.128.0.3`.
- Startup script provisions two venvs at `/opt/aelp/venvs` (`aelp-heavy`, `aelp-light`) with packages: numpy, numba, scipy, pandas, scikit-learn, gymnasium, tensorflow 2.18, tensorflow-probability 0.24, RecSim-NG 0.2, lifelines, SQLAlchemy, redis, Google Cloud clients.
- No long-running RL processes detected (checked `ps aux`). Directory `/opt/aelp` currently contains only the virtualenvs; no cloned repos yet.

Recommendations / Next Steps
1. Clone the RL simulation repo (AELP/AELP2) under `/opt/aelp/work` and configure systemd/tmux service for orchestrator.
2. Copy `.env` / secrets via `gcloud secrets versions access` or scp from controller, ensuring Ad Library/API tokens are not stored in plaintext.
3. Configure firewall: restrict SSH to VPN/IP allowlist; ensure only AELP control plane accesses port 22.
4. Add logs directory `/var/log/aelp` for simulator runs (startup script references it but directory absent due to missing creation step).
5. Document data egress: set up BQ dataset sink for RL outputs (e.g., `gaelp_training.rl_sim_results`).
