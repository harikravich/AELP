Here’s a compact context snapshot you can use to rehydrate next session.

Snapshot — 2025-09-16

- Project/Dataset: aura-thrive-platform.gaelp_training
- Data sources (now): Google Ads + GA4; (next) Meta + Impact.com
- KPI (SoU): GA4 enrollments blend (ga4_enrollments_daily)
  CAC = SUM(ads_campaign_performance.cost_micros/1e6) / SUM(ga4_enrollments_daily.enrollments)

Baselines (same-window aggregates)

- 45‑day blended CAC: ≈ $199.16; spend/day ≈ $191.6k; enroll/day ≈ 962
- Search impression share (14d, weighted): ≈ 17.1%
- CAC by window (14/30/60/90d): ≈ $192 / $260 / $171 / $141

MMM

- Latest run: 2024‑09‑01..2025‑09‑15 (AELP2_MMM_USE_KPI=1)
- Headroom: proposed_daily_budget ≈ $366.3k; expected_cac ≈ $8.61; uncertainty median band ≈ 7.9%
- Tables: mmm_curves, mmm_allocations (dataset gaelp_training)

RL / Orchestrator

- Bandit decisions + orchestrator proposals logged (last 30d: 21 proposals) — shadow/HITL only
- Approvals queue populated; publishing stays PAUSED + allowlist + validateOnly

Freshness (as of now)

- Ads: 2025‑09‑15
- GA4 daily (aggregates): 2025‑09‑07
- GA4 lagged attribution: 2025‑09‑16
- Training (episodes ts): 2025‑09‑11

Guardrails

- dataset prod=read‑only; sandbox=writes (cookie aelp-dataset)
- Google Ads allowlist (GOOGLE_ADS_ALLOWED_CUSTOMERS); publish PAUSED; HITL approvals required

What to click (rehydrate quickly)

- Backstage → check /api/control/status and /api/bq/freshness
- Executive → KPI = GA4 enrollments blend; confirm CAC/enroll/day
- Spend Planner → view MMM curves/headroom; run What‑If
- Creative Center → review bandit decisions; Approvals → publish PAUSED; enable in Ads

Useful commands

- Bring up dashboard: `bash AELP2/scripts/aelp2ctl.sh start`
- Background jobs: `bash AELP2/scripts/run_long_jobs.sh` then `bash AELP2/scripts/check_long_jobs.sh`
- MMM (KPI‑only): `AELP2_MMM_USE_KPI=1 python3 AELP2/pipelines/mmm_service.py --start 2024-09-01 --end 2025-09-15`
- Bandit (shadow): `python3 -m AELP2.core.optimization.bandit_service --lookback 30` and `python3 -m AELP2.core.optimization.bandit_orchestrator --lookback 30`
- Pacing tie‑outs: `python3 AELP2/scripts/pacing_reconcile_ga4.py`
- Dataset mode: `curl -X POST 'http://127.0.0.1:3000/api/dataset?mode=sandbox'`
- Status: `curl -s 'http://127.0.0.1:3000/api/control/status' | jq`

Pending access (next)

- Meta Ads: META_ACCESS_TOKEN (system user or 60‑day), META_ACCOUNT_ID=192357189581498
- Impact.com: IMPACT_ACCOUNT_SID, IMPACT_AUTH_TOKEN
- GA4 export (events_*): grant Viewer on ga360-bigquery-datashare.analytics_308028264 to VM SA 556751870393-compute@developer.gserviceaccount.com, or link export into aura-thrive-platform

Notes

- Executive KPI source can be toggled: Ads conversions vs GA4 enrollments via aelp-kpi-source cookie
- API/UI ports: 3000 (Next API), 8080 (external UI). All creative publishes are PAUSED with CID allowlist

Docs worth opening

- SoU: AELP2/docs/SOU_2025-09-16.md
- Master plan: AELP2/docs/MASTER_ARCHITECTURE_AND_TODO.md
- Env catalog: AELP2/docs/ENV_VARS_CATALOG.md
