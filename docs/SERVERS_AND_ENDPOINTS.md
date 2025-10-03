# Servers and Endpoints (Project: aura-thrive-platform)

Updated: 2025-10-02

## Compute/VMs

- merlin-l4-1 (GPU VM)
  - Zone: us-central1-c
  - Type: g2-standard-16 (L4 GPU)
  - Role: training/scoring, feature extraction (CLIP/YOLO), daily jobs
  - SSH: `gcloud compute ssh <user>@merlin-l4-1 --zone us-central1-c`
  - Code: `/home/harikravich_gmail_com/AELP`
  - Venv: `/srv/aelp/.venv`
  - Env: `/srv/aelp/.env.local` (autoloads for `aelp` group)

- thrive-backend
  - Zone: us-central1-a
  - Type: n1-standard-16
  - Role: backend services host (nginx + containers); staging/prod API endpoints
  - SSH: `gcloud compute ssh <user>@thrive-backend --zone us-central1-a`
  - Notes: running nginx and Docker; has `~/AELP` directory (lightweight)

- aelp-sim-rl-1
  - Zone: us-central1-a
  - Type: n2-standard-16
  - Role: simulation/RL jobs (CPU); batch runners
  - SSH: `gcloud compute ssh <user>@aelp-sim-rl-1 --zone us-central1-a`
  - Notes: no `~/AELP` by default; mount or sync as needed

## Cloud Run Services (us-central1)

- aelp2-dashboard
  - URL: https://aelp2-dashboard-oz5ujfs5oa-uc.a.run.app
  - Role: internal dashboard (AELP2/apps/dashboard)

- thrive-simple-backend
  - URL: https://thrive-simple-backend-oz5ujfs5oa-uc.a.run.app
  - Role: Thrive backend API (lightweight)

- thrive-ui-universal
  - URL: https://thrive-ui-universal-oz5ujfs5oa-uc.a.run.app
  - Role: Thrive UI (universal renderer)

- thrive-mcp-service
  - URL: https://thrive-mcp-service-oz5ujfs5oa-uc.a.run.app
  - Role: MCP service for platform connectors

## Storage (GCS)

- gs://gaelp-model-checkpoints-hariravichandran (primary)
  - balance-assets/, balance-landers/, checkpoints/, coldstart/, creative/, creative_pipeline/, creatives/
- gs://gaelp-model-checkpoints-1755753776 (present)
- gs://aelp-repo-drop-556751870393 (scratch)
- gs://run-sources-aura-thrive-platform-us-central1 (Cloud Run build sources)

## BigQuery (typical)

- Datasets: `gaelp_training`, `gaelp_users` (see infrastructure/bigquery)
- Views: campaign_performance_summary, agent_performance_comparison, …

## Secrets

- Secret Manager: `AELP_DOTENV` (env for CLI/VM); access granted to Bill & Isotta

## Useful Commands

```bash
# Inspect Cloud Run services
gcloud run services list --region us-central1 --project aura-thrive-platform

# GCS inventory
GOOGLE_CLOUD_PROJECT=aura-thrive-platform ./scripts/checks/list_gcs_inventory.sh

# BigQuery inventory
GOOGLE_CLOUD_PROJECT=aura-thrive-platform ./scripts/checks/list_bq_inventory.sh
```
