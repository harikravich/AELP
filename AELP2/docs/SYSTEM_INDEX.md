# AELP + AELP2 System Index (Thorough Inventory)

Purpose: a single navigable map of what exists today across AELP (legacy) and AELP2, with code paths you can spot‑check.

Scope covered in this index
- Dashboard (Next.js) pages and APIs
- Pipelines (Ads/GA4 ingestion, attribution, MMM, bandits/RL, reach, value, audience, DQ)
- Core services (optimization/orchestration/safety)
- Scripts and ops
- Open questions and known gaps

## Dashboard (Next.js App Router)
- Project: `AELP2/apps/dashboard`
- Pages (SSR):
  - `/exec` → `src/app/exec/page.tsx`
  - `/finance` → `src/app/finance/page.tsx`
  - `/growth-lab` → `src/app/growth-lab/page.tsx`
  - `/creative-center` → `src/app/creative-center/page.tsx`
  - `/journeys` → `src/app/journeys/page.tsx`
  - `/training-center` → `src/app/training-center/page.tsx`
  - `/auctions-monitor` → `src/app/auctions-monitor/page.tsx`
  - `/control` → `src/app/control/page.tsx`
  - `/canvas` → `src/app/canvas/page.tsx`
  - `/onboarding` → `src/app/onboarding/page.tsx`
- APIs (server only, BigQuery on server): `src/app/api/**`
  - BQ reads: e.g., `bq/kpi`, `bq/freshness`, `bq/mmm/*`, `bq/creatives`, `bq/ga4/*`, `bq/journeys/*`
  - Control stubs (HITL, flag‑gated): `control/ga4-ingest`, `control/ga4-attribution`, `control/ads-ingest`, `control/apply-canary`, `control/value-upload/*`, etc.
  - A/B logging: `ab/exposure/route.ts` (writes), reads at `bq/ab-*` (expects tables)
  - Creative preview: `ads/creative/route.ts` (Google Ads read via google-ads-api)
- Libs
  - Dataset selection: `src/lib/dataset.ts` (cookie `aelp-dataset`)
  - BigQuery serializer: `src/lib/bigquery-serializer.ts` + client wrapper `bigquery-client.ts`

## Pipelines (AELP2/pipelines)
Key producers/consumers and code paths:
- Google Ads → BQ ingestion
  - `google_ads_mcc_to_bq.py`, `*_to_bq.py` (ad_performance, adgroups, keywords, geo_device, conversion_actions, conversion_stats)
- GA4 → BQ and lag attribution
  - `ga4_to_bq.py`, `ga4_lagged_attribution.py`, `ga4_permissions_check.py`
- Views creation and channel rollups
  - `create_bq_views.py`, `create_channel_views.py`
- MMM / portfolio
  - `mmm_service.py`, `mmm_lightweightmmm.py`, Robyn: `robyn_runner.py`, validator `robyn_validator.py`
- Channel attribution validator
  - `channel_attribution_r.py`, `channel_attribution_runner.py`, `channel_attribution/`
- Bandits / RL
  - `creative_bandit_head.py`, `creative_ab_planner.py`, `offpolicy_eval.py`, `rl_policy_hints_writer.py`
- Reach & recommendations
  - `youtube_reach_planner.py`, `google_recommendations_scanner.py`, `opportunity_scanner.py`
- Value uploads
  - `value_bridge.py`, `upload_google_offline_conversions.py`, `upload_meta_capi_conversions.py`
- Journeys / uplift / segments
  - `journeys_populate.py`, `journey_path_summary.py`, `propensity_uplift.py`, `uplift_eval.py`, `segments_to_audiences.py`
- DQ / monitoring / safety
  - `check_data_quality.py`, `permissions_check.py`, `ops_alerts_stub.py`, `quality_signal_daily.py`, `trust_gates_evaluator.py`

## Core (AELP2/core)
- Optimization
  - `optimization/bandit_service.py`, `optimization/pmax_bandit_service.py`, `optimization/budget_orchestrator.py`, `optimization/audience_bandit_service.py`
- Orchestration
  - `orchestration/production_orchestrator.py`, `orchestration/budget_broker.py`, `orchestration/policy_enforcer.py`
- Intelligence / attribution
  - `intelligence/reward_attribution.py`
- Ingestion helpers
  - `ingestion/bq_loader.py`
- Safety & flags
  - `safety/feature_flags.py`, `safety/feature_gates.py`, `safety/hitl.py`

## Scripts & Ops (AELP2/scripts, AELP2/ops)
- Deploy dashboard to Cloud Run: `deploy_dashboard.sh`, `deploy_dashboard_cloudrun.sh`
- Jobs orchestration: `nightly_jobs.sh`, `run_*` scripts (ads ingest, GA4 lag, fidelity, DQ, attribution)
- Canary ops: `apply_google_canary.py`, `canary_rollback.py`, `snapshot_canary_budgets.py`
- Creative apply (proposal log): `apply_google_creatives.py` (no live mutations yet)
- E2E orchestrator (demo path): `e2e_orchestrator.py`
- Ops scripts: `ops/morning_report.sh`, `ops/scheduler_stub.py`

## Legacy (AELP) – Major Modules To Catalog (kept for reference and reuse)
- RL training/orchestration: `training_orchestrator/*` (agents, online learner, delayed rewards/conversions, safety monitor)
- AuctionGym (simulation): `auction-gym/src` (Agent, Auction, Bidder, models)
- Core (legacy): `core/*` (env/simulator/calibration, platform adapter)
- Attribution/GA4 analysis, cross‑account attribution: various top‑level scripts and docs (to be cross‑mapped to AELP2 equivalents)
- Production checkpoints: `gaelp_production_checkpoints/*` (models/policies)

## Known Gaps (as of 2025‑09‑09)
- Creative publishing: missing real publisher for Google Ads (RSA/PMax) and YouTube video attach; proposals logged only.
- A/B framework: exposures route exists; no assignment service/table; no unified results view.
- Halo: no `halo_*` tables; geo‑rotation tooling not wired.
- LP Studio: no block builder + publish workflow in Next.js; LP tests tables absent.
- Audience export: adapters present (core/data/*), but dashboard control and logs are partial.
- Serializer usage inconsistencies in a few pages (direct `new BigQuery` without wrapper).

---

This index complements: `AELP2/docs/MASTER_ARCHITECTURE_AND_TODO.md`, `PROJECT_PLAN_VIEW.md`, `DASHBOARD_TODO_MASTER.md`.
