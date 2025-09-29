# Feature Coverage Matrix (AELP + AELP2)

Legend: [OK]=present & working, [P]=partial, [M]=missing, [N/A]=not applicable

## Data & Ingestion
- Ads → BigQuery: [OK]
  - Files: `AELP2/pipelines/google_ads_*_to_bq.py`, `google_ads_mcc_to_bq.py`, `google_ads_to_bq.py`
- GA4 → BigQuery: [OK]
  - Files: `AELP2/pipelines/ga4_to_bq.py`, `ga4_permissions_check.py`
- GA4 lag attribution: [OK]
  - Files: `AELP2/pipelines/ga4_lagged_attribution.py`, view `ga4_lagged_daily` via `create_bq_views.py`
- Views (daily rollups): [OK]
  - Files: `AELP2/pipelines/create_bq_views.py`, `create_channel_views.py`

## Modeling & Learning
- MMM (LightweightMMM) daily/weekly: [P]
  - Files: `AELP2/pipelines/mmm_service.py`, `mmm_lightweightmmm.py` (needs schedule + outputs surfaced)
- MMM validator (Robyn): [P]
  - Files: `AELP2/pipelines/robyn_runner.py`, `robyn_validator.py` (containers present)
- Bandits/RL (TS): [P]
  - Files: `AELP2/core/optimization/bandit_service.py`, `creative_bandit_head.py` (posteriors persist missing)
- Attribution validator (Markov/Shapley): [P]
  - Files: `AELP2/pipelines/channel_attribution_r.py`, `channel_attribution_runner.py`
- Uplift/propensity: [P]
  - Files: `propensity_uplift.py`, `uplift_eval.py` (results not surfaced in UI)

## Experiments & Flags
- A/B definitions table/view: [P]
  - Files: reads `bq/ab-*`; tables not always present; `ab_experiments_daily` in `create_bq_views.py`
- Assignments & exposures: [P]
  - Files: `api/ab/exposure` writes; assignment service [M]
- Flags (server): [P]
  - Files: `core/safety/feature_flags.py`, `feature_gates.py` (SDK integration [M])

## Creative & Publishing
- Creative preview (read): [OK]
  - Files: `apps/dashboard/src/app/api/ads/creative/route.ts`
- Creative proposals log: [OK]
  - Files: `scripts/apply_google_creatives.py` (shadow log only)
- Creative publisher (Google Ads RSA/PMax): [M]
  - Needed: `pipelines/publish_google_creatives.py`, queue/log tables, API endpoints
- Video pipeline (YouTube upload/attach): [M]
  - Needed: YouTube Data API integration + Ads asset attach

## Landing Pages
- LP metrics & candidates: [P]
  - Files: `api/bq/lp-ab` (expects `lp_ab_candidates`), funnel views [M]
- LP Studio (blocks + publish + % routing): [M]
  - Needed: pages, APIs, tables `lp_tests`, `lp_block_metrics`

## Journeys & Halo
- GA4 device/channel & lag: [OK]
  - Files: `api/journeys/page.tsx`, `ga4_*` tables
- Halo (brand‑lift, interference): [M]
  - Needed: `halo_experiments`, `halo_reads_daily`, GeoLift job, interference scores

## Audience & Exports
- Segment scoring: [P]
  - Files: `segment_scores_daily` expected; `segments_to_audiences.py`
- Audience sync (Google/Meta/TikTok): [P]
  - Files: adapters in `core/data/*` (UI controls/logs [M])

## Value & Offline Conversions
- Value bridge + uploads: [OK/P]
  - Files: `value_bridge.py`, `upload_google_offline_conversions.py` (gated), `value_uploads_log`
  - UI controls exist; need clearer status surface

## Dashboard Quality
- BigQuery serialization: [P]
  - Files: serializer present; some pages use raw client → fix
- Freshness API: [OK]
  - Files: `api/bq/freshness`
- Exec Summary (LLM) + proof drawers: [P]
  - Files: chat route present; OPENAI key missing path & UI integration [P]

## Safety & HITL
- Gating & approval flow: [P]
  - Files: `safety/hitl.py`, control endpoints; need unified approval queue UI
- Change caps (per change/day): [P]
  - Files: enforced in canary scripts; surface in UI [M]

## Deploy & Ops
- Cloud Build/Run deploys: [OK]
  - Files: `apps/dashboard/Dockerfile*`, `cloudbuild*.yaml`, `scripts/deploy_dashboard.sh`
- Ops scripts: [OK]
  - Files: `nightly_jobs.sh`, `run_*`, `ops/morning_report.sh`

---

Summary of missing pillars to reach test→learn→scale in prod
- Creative publisher (paused by default) + queue/log tables + UI buttons
- A/B assignment service/table; unified results views with SRM & decision rules
- Explore cells + bandit posteriors persistence & UI
- Halo (GeoLift/interference) tables + weekly job; surface in ramp decisions
- LP Studio (blocks/publish/funnel) + tables
- Audience sync controls/logs (shadow → live)
- LLM summaries/variants wired with graceful off-state

