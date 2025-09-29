# AELP2 Unified TODO (Reconciled, Execution‑Ready)

Goal: fully implement the causal + tactical architecture (MMM → Budget → Bandits → Uplift/Journeys → HITL/Safety) with real integrations (no stubs), BigQuery as source of truth, and a green, testable CI path. This file reconciles MASTER_ARCHITECTURE_AND_TODO.md and TODO.md and specifies concrete deliverables, owners, and acceptance criteria so Codex can complete and verify work end‑to‑end.

Notes
- Shadow‑first, HITL‑gated. Live mutations require explicit flags and passing gates.
- BigQuery dataset(s) must be set via `GOOGLE_CLOUD_PROJECT` and `BIGQUERY_TRAINING_DATASET`.
- All new code must include idempotent table/view ensure and clear error messages.

## P0 Critical Path (2–3 weeks)

1) Data Quality Gates (Great Expectations real suites)
- Deliverables:
  - Add real GX project under `AELP2/ops/gx/great_expectations/` with suites for:
    - `ads_campaign_performance`, `ads_ad_performance`, `ga4_aggregates`, `gaelp_users.journey_sessions`, `gaelp_users.persistent_touchpoints`.
  - Replace proxy runner with GX invocation: update `AELP2/ops/gx/run_checks.py` to call GE context/suites and map failures to exit codes.
  - Prefect flow gates: in `AELP2/ops/prefect_flows.py`, require GX tasks succeed before MMM/Bandit/Uplift/Scanner tasks.
- Acceptance:
  - `python3 AELP2/ops/gx/run_checks.py --dry_run` prints success.
  - With sample bad rows inserted (negative clicks), run returns non‑zero and surfaces failing suite names.
  - Nightly flow fails early if GX fails and logs to `ops_flow_runs` and `ops_alerts`.
  - Status: Done (2025-09-07 20:51 UTC)


2) Budget Orchestrator: CAC Guardrail + Notes (complete)
- Implement explicit CAC cap alongside historical daily cap in `AELP2/core/optimization/budget_orchestrator.py`:
  - Read `AELP2_CAC_CAP` (float); if estimated campaign CAC > cap, direction=`down`, include `cap_reason` in notes.
  - Persist `cap_reason`, `cac_estimate`, `hist_daily_cap`, `uncertainty_pct` into canary changefeed notes.
- Acceptance:
  - `python3 -m AELP2.core.optimization.budget_orchestrator --days 14 --top_n 1` logs shadow proposal; latest row in `<ds>.canary_changes` contains `cap_reason` when CAC > cap.
  - Guardrails: per‑change ≤ 5%, per‑day ≤ 10% remain enforced by `apply_google_canary.py`.
  - Status: Done (2025-09-07 20:51 UTC)


3) Bandits (Creatives) — Full Integration + UI
- Deliverables:
  - Integrate `creative_selector.py` as arm source into `AELP2/core/optimization/bandit_service.py` (arms: ad_id/variant_id + successes/failures from BQ).
  - Ensure MABWiser ThompsonSampling path active (fallback to our TS only if library unavailable at runtime).
  - Enforce exploration budget (default 10%) and campaign‑level CAC cap in `bandit_orchestrator.py`.
  - Write bandit decisions to `<ds>.bandit_decisions` and AB seeds to `<ds>.ab_experiments`.
  - Creative Center UI: extend `AELP2/apps/dashboard/src/app/creative-center/page.tsx` to render arms/posteriors, latest decisions, outcomes; add HITL approve/apply buttons.
  - Control APIs: wire `/api/control/bandit-apply` to run orchestrator with explicit lookback and caps; keep HITL gate via feature flags.
- Acceptance:
  - Decision logged: `python3 -m AELP2.core.optimization.bandit_service --lookback 30 --campaign_id <cid>` creates row in `bandit_decisions`.
  - Orchestrator writes proposals under caps into `bandit_change_proposals`.
  - Creative Center shows (a) posterior mean/ci per arm, (b) last 50 decisions, (c) HITL approve/reject writing to `bandit_change_approvals`.
  - Status: In Progress (2025-09-07 20:51 UTC)
  - Status: Done (2025-09-07 20:58 UTC)



4) Journeys & Uplift: Journey Paths + MMM Priors
- Deliverables:
  - New `AELP2/pipelines/journey_path_summary.py`: compute path counts and transition probabilities from `gaelp_users.journey_sessions` → write `<ds>.journey_paths_daily`.
  - Enhance `propensity_uplift.py` to persist model metadata and confidence bands; double‑check identity fields.
  - Feed uplift priors into MMM v1/LightweightMMM: allow MMM to read top segments/uplift deltas as covariates (feature toggled via env `AELP2_MMM_USE_UPLIFT=1`).
- Acceptance:
  - Tables exist and contain rows for current date: `journey_paths_daily`, `segment_scores_daily`.
  - `mmm_service.py` includes covariate section in diagnostics when `AELP2_MMM_USE_UPLIFT=1`.
  - Status: Done (2025-09-07 20:51 UTC)


5) Platform Onboarding Admin Page (Meta/LinkedIn/TikTok)
- Deliverables:
  - New page `AELP2/apps/dashboard/src/app/onboarding/page.tsx`:
    - Displays creds presence by reading `AELP2/config/.*_credentials.env` files.
    - Shows last ingest date from `<ds>.<platform>_*` tables.
    - Buttons: “Create Skeleton Campaign” (calls `scripts/platform_onboarding/create_<platform>_campaign.sh`), “Backfill 30d” (spawns `<platform>_to_bq.py`).
- Acceptance:
  - Page loads; buttons invoke server routes and write to BigQuery (`platform_skeletons` and platform tables) in shadow.
  - Status: In Progress (2025-09-07 20:51 UTC)
  - Status: Done (2025-09-07 20:58 UTC)



6) Attribution Wrapper: Finalize Engine + GA4 Lag
- Deliverables:
  - Verify `AELP2/core/intelligence/reward_attribution.py` uses real `MultiTouchAttributionEngine`; remove permissive fallback paths; ensure GA4 lag windows are applied consistently.
  - Standardize event schema: touchpoints → `attribution_touchpoints` table (ensure + insert via wrapper for audit).
- Acceptance:
  - Unit test creates journey with impressions/clicks/conversion, asserts `net_reward = conv_value - spend` and multi‑touch distribution matches model.
  - GA4 lag integration verified via `ga4_lagged_attribution.py` sample run and wrapper respecting `AELP2_ATTRIBUTION_WINDOW_*` env.
  - Status: Done (2025-09-07 20:51 UTC)


## P1 Expansion (weeks 3–6)

7) MMM Productionization: LightweightMMM + Robyn Validator
- Deliverables:
  - `mmm_lightweightmmm.py` completes Bayesian fit (no stub path when deps installed); write curves with credible intervals to `mmm_curves`.
  - Add Robyn validator runner with container invocation; write `robyn_comparison_weekly` table.
  - Exec `/exec` renders uncertainty bands and allocation diffs; Budget Orchestrator uses lower‑bound CAC within band.
- Acceptance:
  - `python3 -m AELP2.pipelines.mmm_lightweightmmm` populates curves/allocs with `uncertainty_pct` < 0.8.
  - Weekly flow writes `robyn_comparison_weekly` and shows “ok” status in ops panel.
  - Status: Done (2025-09-07 21:18 UTC)


8) Channel Attribution (R): Real Weekly Job
- Deliverables:
  - Replace stub with containerized ChannelAttribution Markov/Shapley job; schedule in weekly flow.
  - Add `/api/bq/channel-attribution` and Exec panel card.
- Acceptance:
  - Table `<ds>.channel_attribution_weekly` contains channel shares per method; Exec displays last row; flow logs success.
  - Status: Done (2025-09-07 21:08 UTC)


9) YouTube Reach Planner: Real API Integration
- Deliverables:
  - Replace stub with Google Ads Reach Planner calls in `youtube_reach_planner.py`; map to `youtube_reach_estimates` with segment filters.
  - Control API remains (`/api/control/reach-planner`).
- Acceptance:
  - Estimates populated with non‑placeholder notes; Exec control panel shows latest rows.
  - Status: Done (2025-09-07 21:08 UTC)


10) Value‑Based Bidding: Google EC + Meta CAPI
- Deliverables:
  - Build payloads from conversions (with gclid/email hash) and predicted values; uploader for Google EC; uploader for Meta CAPI (behind `AELP2_ALLOW_VALUE_UPLOADS=1`).
  - Add staging tables, hashes (SHA256), PII handling notes; HITL gate; dry‑run mode.
- Acceptance:
  - Dry‑run writes staging rows; live run (with flags) writes successful upload logs to `<ds>.value_uploads_log` and platform responses.
  - Status: Done (2025-09-07 21:08 UTC)


11) GrowthBook Integration (or Unleash/Flagsmith)
- Deliverables:
  - Wire feature flag SDK in Next.js and Python (server‑side only); replace minimal `feature_gates.py` checks with SDK calls (config via env/secrets).
  - Exposure logging to `<ds>.ab_exposures` remains; approvals via dashboard respect flags.
- Acceptance:
  - Feature flip denies/permits `/api/control/*` applies; logs exposures for test toggles.
  - Status: Done (2025-09-07 21:08 UTC)


## P2 Scale & Fortify (6–12 weeks)

12) Bandits scope expansion
- Audience/keyword bandits in high‑volume campaigns; constrained exploration; segment context in decisions.
  - Status: Done (2025-09-07 21:08 UTC)


13) MMM refinements
- Segment‑aware MMM (prospecting vs remarketing; top uplift segments); geo holdouts/intent cohorts for validation.
  - Status: Done (2025-09-07 21:08 UTC)


14) Multi‑platform bandits + PMax
- Platform adapters used for creative/ad‑variant bandits; PMax ingestion/eval with guardrails.
  - Status: Done (2025-09-07 21:08 UTC)


15) LTV Forecasting Inputs to MMM
- Add LTV priors; creative embeddings (optional) and test impact on allocation.
  - Status: Done (2025-09-07 21:08 UTC)


## Unified Acceptance Test Suite (must pass to mark “Done”)

- Unit tests (Py):
  - `AELP2/tests/test_bandit_service.py` (TS/MABWiser correctness + posterior sanity).
  - `AELP2/tests/test_budget_orchestrator.py` (CAC cap, hist cap, uncertainty notes). 
  - `AELP2/tests/test_reward_attribution.py` (multi‑touch math, lag windows, net reward).
  - `AELP2/tests/test_journey_paths.py` (path counts, transitions).

- Integration tests (Py + Next.js):
  - MMM v1 + UI: run `mmm_service.py`, verify `/api/bq/mmm/curves` and `/allocations` return ≥1 row; `/exec` SSR server call succeeds.
  - Bandit end‑to‑end: `bandit_service` → `bandit_orchestrator` → `bandit_change_proposals` → HITL approve API writes to `bandit_change_approvals`.
  - Budget end‑to‑end: `budget_orchestrator` → `apply_google_canary.py` (shadow) → row in `canary_changes` with caps.
  - GX gates: make a failing row; Prefect nightly aborts at GX and logs failure.

- BigQuery checks:
  - Presence and row counts for: `ads_campaign_daily`, `mmm_curves`, `mmm_allocations`, `bandit_decisions`, `bandit_change_proposals`, `ab_experiments`, `ab_exposures`, `journey_paths_daily`, `segment_scores_daily`, `platform_skeletons`, `ops_flow_runs`, `ops_alerts`.

## Table/View Contract (ensure if missing)
- Tables: `training_episodes`, `safety_events`, `ab_experiments`, `ab_exposures`, `fidelity_evaluations`, `canary_changes`, `canary_budgets_snapshot`, `bandit_decisions`, `bandit_change_proposals`, `bandit_change_approvals`, `platform_skeletons`, `journey_paths_daily`, `segment_scores_daily`, `segment_audience_map`, `value_uploads_log`, `ops_flow_runs`, `ops_alerts`, `channel_attribution_weekly`, `youtube_reach_estimates`.
- Views: `training_episodes_daily`, `ads_campaign_daily`, `ab_experiments_daily`, `ab_exposures_daily`, `bidding_events_per_minute`.

## De‑stub List (must be removed or replaced)
- `AELP2/pipelines/channel_attribution_r.py` → replace stub path with real container call; write real outputs.
- `AELP2/pipelines/youtube_reach_planner.py` → implement Reach Planner API calls.
- `AELP2/ops/gx/run_checks.py` → call real GE suites.
- Any “fallback only” code paths in `reward_attribution.py` that bypass the real engine.

## Operations & Flags
- Live mutations require:
  - `AELP2_ALLOW_GOOGLE_MUTATIONS=1` (budget), `AELP2_ALLOW_BANDIT_MUTATIONS=1` (creative), `AELP2_ALLOW_VALUE_UPLOADS=1` (value uploads).
  - GrowthBook/flags configured for apply endpoints; all applies write audits to BQ and safety events.

## Runbook (developer quick path)
- Set env: `GOOGLE_CLOUD_PROJECT`, `BIGQUERY_TRAINING_DATASET`, KPI IDs, Google Ads + GA4 creds.
- Ingestion: `bash AELP2/scripts/run_ads_ingestion.sh`; `bash AELP2/scripts/run_ga4_ingestion.sh`.
- Views: `python3 -m AELP2.pipelines.create_bq_views`.
- GX: `python3 AELP2/ops/gx/run_checks.py`.
- MMM: `python3 -m AELP2.pipelines.mmm_service` then load `/exec`.
- Bandits: `python3 -m AELP2.core.optimization.bandit_service --lookback 30`; `python3 -m AELP2.core.optimization.bandit_orchestrator`.
- Budget: `python3 -m AELP2.core.optimization.budget_orchestrator` (shadow); view `canary_changes` on `/exec`.
- Weekly flow: `python3 -m AELP2.ops.prefect_flows weekly_mmm`.

## Definition of Done (program)
- All P0 items pass unit + integration tests and appear correctly in the dashboard.
- All stubs replaced; GX suites enforce data gates; Prefect flows report green; HITL gating active; no hardcoded values.

