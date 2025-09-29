# AELP2 Rehydrate Context (Snapshot)

Last updated: 2025-09-08

This page is a compact, self‑contained bundle you can use to restore the working state on a new or cleaned machine. Keep it in the repo; do not commit real secrets.

## Snapshot (Env + Flags)

- Project: aura-thrive-platform
- Training dataset: gaelp_training
- Users dataset: gaelp_users
- GE flags: `AELP2_GX_USE_BQ=1`, `AELP2_GX_CLICK_IMP_TOLERANCE=0.05`, `AELP2_GX_MAX_VIOLATION_ROWS_PCT=0.10`
- Containers (optional): `AELP2_ROBYN_ALLOW_RUN=1`, `AELP2_CA_ALLOW_RUN=1`
- Live toggles (keep 0 by default): `AELP2_ALLOW_GOOGLE_MUTATIONS`, `AELP2_ALLOW_VALUE_UPLOADS`, `AELP2_VALUE_UPLOAD_DRY_RUN`
- Gmail Ads mapping: if `GMAIL_CLIENT_ID/SECRET/REFRESH_TOKEN` are present, they override `GOOGLE_ADS_*`; `GMAIL_CUSTOMER_ID` maps to both `GOOGLE_ADS_CUSTOMER_ID` and `GOOGLE_ADS_LOGIN_CUSTOMER_ID` when needed.
- GA4 SA JSON (optional): `GA4_SERVICE_ACCOUNT_JSON=/abs/path/to/ga4_sa.json` → sets `GOOGLE_APPLICATION_CREDENTIALS`.
 - Chat/LLM (optional):
   - `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o` (default lane for analysis/UI polish)
   - Optional self‑hosted: `OPENAI_BASE_URL=http://<vm-ip>:8000/v1`, `OPENAI_MODEL=<model>` (vLLM/TGI)
   - (Optional) Anthropic lane: `ANTHROPIC_API_KEY` if you enable a design/provider router later.

## Install/Deps

- Python: `python3 -m pip install --upgrade pip`
- Python deps: `python3 -m pip install google-cloud-bigquery db-dtypes great_expectations mabwiser prefect google-ads pyarrow pandas`
- Node (optional for dashboard): Node 18+, then `npm install --no-audit --no-fund && npm run build`
- Docker (optional): docker installed and daemon running

## One‑Command Orchestrator

- Safe full run (auto‑detect creds, GE on BQ, containers only if enabled):
  - `set -a; source .env; set +a; python3 AELP2/scripts/e2e_orchestrator.py --auto`
- With containers: `export AELP2_ROBYN_ALLOW_RUN=1 AELP2_CA_ALLOW_RUN=1` then run the command above
- Live actions (only when ready): `export AELP2_ALLOW_GOOGLE_MUTATIONS=1 AELP2_ALLOW_VALUE_UPLOADS=1 AELP2_VALUE_UPLOAD_DRY_RUN=0`

## Files Added/Updated (core)

- `AELP2/scripts/e2e_orchestrator.py` (single‑shot runner; auto‑loads `.env` + gmail creds; logs to `ops_flow_runs`)
- `AELP2/scripts/preflight.py` (BQ contract ensures + health checks)
- `AELP2/ops/gx/run_checks.py` (GE gate; BQ read with tolerances; exit 1 on fail)
- `AELP2/core/optimization/budget_orchestrator.py` (CAC cap, conservative CAC; Gmail mapping; shadow proposal notes)
- `AELP2/core/optimization/bandit_service.py` (decisions + AB seed)
- `AELP2/core/optimization/bandit_orchestrator.py` (exploration + CAC cap; JSON default=str fix)
- `AELP2/core/optimization/audience_bandit_service.py` (segments → bandit_decisions)
- `AELP2/core/optimization/pmax_bandit_service.py` (PMax → bandit_decisions)
- `AELP2/pipelines/journey_path_summary.py` (writes `journey_paths_daily`)
- `AELP2/pipelines/propensity_uplift.py` (`segment_scores_daily` with metadata/CI)
- `AELP2/pipelines/ltv_priors.py` (writes `ltv_priors_daily`)
- `AELP2/pipelines/mmm_service.py` + `mmm_lightweightmmm.py` (curves/allocations; uplift covariates in diagnostics)
- `AELP2/pipelines/robyn_validator.py` (`mmm_validation` + comparison table writer)
- `AELP2/pipelines/channel_attribution_r.py` (container JSON parse + write; diagnostic fallback)
- `AELP2/pipelines/upload_google_offline_conversions.py` (real API path behind flags)
- `AELP2/pipelines/upload_meta_capi_conversions.py` (real Graph API path behind flags)
- `AELP2/adapters/google_enhanced_conversions.py` | `meta_capi.py` (SHA256 hashing payload builders)
- `AELP2/apps/dashboard/src/lib/dataset.ts` (reads datasets from env; no hardcoded training/sandbox)
- `AELP2/docs/SECRETS_REQUIRED.md`, `AELP2/docs/SETUP_BLOCKERS.md` (what to set; how to install Docker)

## BigQuery Tables/Views to Expect

- Tables: `mmm_curves`, `mmm_allocations`, `bandit_decisions`, `bandit_change_proposals`, `ab_experiments`, `canary_changes`, `journey_paths_daily`, `segment_scores_daily`, `ltv_priors_daily`, `value_uploads_log`, `channel_attribution_weekly`, `mmm_validation`, `robyn_comparison_weekly`, `ops_flow_runs`, `ops_alerts`, `attribution_touchpoints`
- Views: `training_episodes_daily`, `ads_campaign_daily`, `ab_exposures_daily`, `bidding_events_per_minute`

## Dashboard Quickstart

- Dev/preview (no Cloud Build):
  - `bash AELP2/scripts/dev_dashboard_local.sh` → installs, builds, starts Next on 3001
  - Nginx reverse proxy script available to expose port 80 (see scripts)
- Dataset switcher:
  - Header toggle “Sandbox/Prod” sets cookie and reloads; pages/APIs must read dataset from cookie (not env). Freshness badges call `/api/bq/freshness` and show latest dates per table.
- ChatOps:
  - Natural‑language KPI/GA4 questions → dynamic SQL planning with safe allowlist; returns `rows` and `viz` chart spec when possible.
  - Commands: `/help`, `/mmm`, `/creative`, `/sql`, `/run ga4_ingest|ads_ingest only=<CID>`.
  - Pin charts to Canvas (BQ: `gaelp_users.canvas_pins`) and add them from the Canvas selector.
- Known gaps to finish:
  - Finish dataset wiring for all `/api/bq/**` routes; unify visuals and empty states across pages; add skeletons/toasts broadly.

## Rehydrate Steps (clean machine)

1. Clone repo; `cd` into it
2. `set -a; source .env; set +a` (ensure `.env` has your real values; do not check secrets into git)
3. Python setup:
   - `python3 -m pip install --upgrade pip`
   - `python3 -m pip install google-cloud-bigquery db-dtypes great_expectations mabwiser prefect google-ads pyarrow pandas`
4. (Optional) Docker setup if you want container jobs; then:
   - `export AELP2_ROBYN_ALLOW_RUN=1 AELP2_CA_ALLOW_RUN=1`
5. (Optional) GA4 SA file:
   - `export GA4_SERVICE_ACCOUNT_JSON=/abs/path/to/ga4_sa.json`
6. One‑shot:
   - `python3 AELP2/scripts/e2e_orchestrator.py --auto`
7. Check BQ tables for new rows; open the dashboard if you built it

## Minimal Cheat Sheet (when memory is tight)

- Run it: `set -a; source .env; set +a; python3 AELP2/scripts/e2e_orchestrator.py --auto`
- Most useful envs to tweak:
  - GE: `AELP2_GX_USE_BQ=1`, `AELP2_GX_CLICK_IMP_TOLERANCE=0.05`, `AELP2_GX_MAX_VIOLATION_ROWS_PCT=0.10`
  - Containers: `AELP2_ROBYN_ALLOW_RUN=1`, `AELP2_CA_ALLOW_RUN=1`
  - Live: `AELP2_ALLOW_GOOGLE_MUTATIONS=1`, `AELP2_ALLOW_VALUE_UPLOADS=1`, `AELP2_VALUE_UPLOAD_DRY_RUN=0`
- Troubleshooting:
  - “db-dtypes” error → `pip install db-dtypes`
  - Ads `USER_PERMISSION_DENIED` → set `GOOGLE_ADS_LOGIN_CUSTOMER_ID` to the correct manager (MCC) that has access to `GOOGLE_ADS_CUSTOMER_ID` (or use Gmail mapping)
  - No GA4 creds → set `GA4_SERVICE_ACCOUNT_JSON`
