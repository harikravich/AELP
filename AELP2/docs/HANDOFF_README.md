# AELP2 Handoff Package

## Environment
- Project: aura-thrive-platform
- Datasets: gaelp_training, gaelp_users; add gaelp_dev/stage/prod per scripts
- Buckets: aelp2-{dev,stage,prod}-{reports,artifacts}
- Region: us-central1

## Access
- Owners: rekha@aura.com, bill.langenberg@aura.com, hari@aura.com, isotta.landi@aura.com, sarah.cherng@aura.com, shuang.hao@aura.com
- Scripts to grant: `AELP2/scripts/gcp/grant_owners.sh`

## Bootstrap (order)
1) `AELP2/scripts/gcp/vars.example.env` → copy to `vars.env` and edit
2) Datasets: `source vars.env && ./AELP2/scripts/gcp/create_datasets.sh`
3) Buckets: `source vars.env && ./AELP2/scripts/gcp/create_buckets.sh`
4) Secrets: `source vars.env && ./AELP2/scripts/gcp/create_secrets.sh`
5) Budgets: `source vars.env && ./AELP2/scripts/gcp/create_budgets.sh`
6) Owners: `source vars.env && ./AELP2/scripts/gcp/grant_owners.sh`

## BigQuery Visual Features
- DDL: `AELP2/scripts/gcp/bq_ddl/*.sql`
- Loaders: `AELP2/tools/load_finals_features_to_bq.py`, `AELP2/tools/load_visual_embeddings_to_bq.py`

## CI/CD
- GitHub Actions: `.github/workflows/*`
- OIDC secrets required: `GCP_WIF_PROVIDER`, `GCP_CI_SA`, `GCP_PROJECT_ID`, `GCP_REGION`, `AELP_REPORTS_DIR`

## Dashboard
- `.env.local.example` provided; deploy via `gcp-deploy.yml` or Cloud Run script.

## Docs
- Build LaTeX v2 with `make docs` or GH Actions workflow `Docs - Build LaTeX v2`.

## Onboarding
- Add users to group (optional) and verify: BQ query, buckets read, dashboard access.

