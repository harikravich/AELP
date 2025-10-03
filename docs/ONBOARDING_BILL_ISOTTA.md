# AELP Onboarding (Bill + Isotta)

This doc gets you productive on this repo with minimal friction. It covers cloning, local setup, secrets, GPU VM access, and common tasks. For a full system map and architecture diagram, see: `docs/SYSTEM_OVERVIEW_BILL_ISOTTA.md`.

## Prereqs
- Python 3.10.x (recommended 3.10.12)
- Node 18+ (only if touching the dashboard)
- `gcloud` CLI installed and logged into the correct Google account
- GitHub access (accept invite): https://github.com/harikravich/AELP/invitations

## Clone + Virtualenv
```bash
git clone git@github.com:harikravich/AELP.git
cd AELP
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
# CPU-friendly torch for quick local runs (GPU: install CUDA build instead)
pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.*'
pip install --prefer-binary -r AELP2/requirements_dev_310.txt
```

## Environment Variables
Two options:

1) Local file (quick start)
- Copy template and fill in values you have:
  ```bash
  cp .env.local.example .env.local
  # then: source .env.local
  ```

2) Secret Manager (preferred for real secrets)
- We’ll store the full `.env` in Google Secret Manager and grant you access.
- Once access is granted, fetch it with:
  ```bash
  # Project is defined in .env.local.example or provided by Hari
  export GOOGLE_CLOUD_PROJECT=aura-thrive-platform
  gcloud secrets versions access latest \
    --secret=AELP_DOTENV \
    --project "$GOOGLE_CLOUD_PROJECT" > .env.local
  source .env.local
  ```

Minimal keys you’ll usually need for read-only tasks:
- `GOOGLE_CLOUD_PROJECT`
- `BIGQUERY_TRAINING_DATASET`
- `META_ACCESS_TOKEN` (read-only scopes: ads_read, read_insights)
- `META_ACCOUNT_ID`

The full template with placeholders is in `./.env.local.example`.

## GPU VM Access (merlin-l4-1)
- Zone: `us-central1-c`
- Code lives at: `~/AELP` (Hari’s home). Your user can log in but may not read Hari’s home by default.

If you need to read artifacts without sudo, we’ll set up a shared path or group access. Proposed options (run by Hari):

Option A — Shared group on `~/AELP` (read/execute):
```bash
sudo groupadd -f aelp
sudo usermod -aG aelp bill
sudo usermod -aG aelp isotta
sudo chgrp -R aelp /home/hariravichandran/AELP
sudo find /home/hariravichandran/AELP -type d -exec chmod 2750 {} +
sudo find /home/hariravichandran/AELP -type f -exec chmod 0640 {} +
```

Option B — Shared workspace at `/srv/aelp` with group sticky bit:
```bash
sudo mkdir -p /srv/aelp
sudo chgrp -R aelp /srv/aelp
sudo chmod -R 2775 /srv/aelp
# Rsync code/artifacts as needed
```

SSH examples:
```bash
gcloud compute ssh merlin-l4-1 --zone us-central1-c
gcloud compute scp --zone us-central1-c merlin-l4-1:~/AELP/artifacts/predictions/ctr_scores_reg.parquet ./
```

## Common Tasks
- Fetch Meta ads + insights (read-only):
  ```bash
  source .env.local
  PYTHONPATH=$PWD python3 tools/meta/fetch_ads_and_insights.py \
    --start 2025-09-01 --end 2025-10-01 --limit 500 --date-preset last_90d
  ```

- Build unified link-CTR table + features + train models:
  ```bash
  PYTHONPATH=$PWD python3 pipelines/data/unify_meta_marketing.py \
    --ads artifacts/meta/ads.csv \
    --ins artifacts/meta/insights.csv \
    --out artifacts/marketing/unified_ctr.parquet

  PYTHONPATH=$PWD python3 pipelines/features/join_creative.py \
    --unified artifacts/marketing/unified_ctr.parquet \
    --creative artifacts/creative/meta_creative_features.parquet \
    --out artifacts/features/marketing_ctr_joined.parquet

  PYTHONPATH=$PWD python3 pipelines/features/enhance_features.py \
    --in artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-30 \
    --out artifacts/features/marketing_ctr_enhanced.parquet

  PYTHONPATH=$PWD python3 pipelines/ctr/train_ctr_creative.py \
    --data artifacts/features/marketing_ctr_enhanced.parquet \
    --out artifacts/models/ctr_creative_enhanced.joblib
  ```

- Score “current-running” ads (unique link CTR):
  ```bash
  PYTHONPATH=$PWD python3 pipelines/ctr/predict_ctr.py \
    --model artifacts/models/ctr_creative_enhanced.joblib \
    --data artifacts/features/marketing_ctr_latest_enhanced.parquet \
    --out artifacts/predictions/ctr_scores_reg.parquet
  ```

## Securely Sharing the .env
- Preferred: Google Secret Manager (single multi-line secret `AELP_DOTENV`). Hari will push the current `.env` and grant you `roles/secretmanager.secretAccessor`.
- Alternative: 1Password vault item shared to your emails.
- Avoid: email/Slack/Docs.

Helper scripts (optional):
- `scripts/secrets/push_env_to_gsm.sh` — create/update the `AELP_DOTENV` secret from a local `.env`
- `scripts/secrets/pull_env_from_gsm.sh` — fetch into `.env.local`

## Questions
Ping Hari in Slack if anything here blocks you. We can promote your IAM as needed (BQ viewer/job user, Storage viewer, Secret Manager accessor).
