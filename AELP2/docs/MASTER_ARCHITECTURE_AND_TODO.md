# AELP / AELP2 Master Architecture & TODO

Goal: materially increase new‑customer volume while holding or reducing blended CAC by adopting a causal + tactical stack — MMM + Uplift/Journey + Bandits + HITL — with RL kept as a simulation/strategy lab. This consolidates what’s in AELP and AELP2, what we’ll use vs. park, the target architecture, concrete jobs, and a phased TODO.

## Executive Summary

- Strategy: Use MMM (weekly, causal truth) to allocate dollars, bandits (daily/hourly) to win micro‑tactics (creatives/audiences/keywords) under strict guardrails, uplift/journey to capture incremental segments and sequencing, HITL for governance. Keep RL (AELP master stack) for what‑ifs; do not drive production control with RL now.
- Why this scales volume and reduces CAC:
  - MMM allocates to true incremental lift (avoids attribution traps).
  - Bandits discover and exploit micro‑wins safely and continuously.
  - Uplift focuses spend on “converts‑because‑of‑ads” segments.
  - HITL + canary + safety gates ensure safe pacing and rollback.

## Current Assets (Repo‑Grounded)

### AELP2 (Production‑oriented; In Use Today)
- Orchestrator: production RL loop with calibration floors, safety/HITL, attribution wrapper, BQ writes
  - `AELP2/core/orchestration/production_orchestrator.py`
- Telemetry → BigQuery (episodes, safety, AB), batching & schema management
  - `AELP2/core/monitoring/bq_writer.py`
- Ads/GA4 pipelines + views: Ads→BQ, GA4→BQ, GA4 lag attribution, view creation
  - `AELP2/pipelines/google_ads_*.py`, `AELP2/pipelines/ga4_to_bq.py`, `AELP2/pipelines/ga4_lagged_attribution.py`, `AELP2/pipelines/create_bq_views.py`
- Fidelity & readiness gates (KPI‑only + KS):
  - `AELP2/pipelines/fidelity_evaluation.py`, `AELP2/scripts/check_canary_readiness.py`
- Shadow Canary (budget‑only) with changefeed & rollback:
  - `AELP2/scripts/apply_google_canary.py`, `AELP2/scripts/rollback_last_canary.py`
- Next.js dashboard & APIs (training, fidelity, safety, creatives, control):
  - `AELP2/apps/dashboard/src/app/api/*`

### AELP (Lab/Simulation; Reviewed for Reuse)
- Master Orchestrator (19–20 components; full sim): `gaelp_master_integration.py`
- Monte Carlo Simulator (parallel worlds): `monte_carlo_simulator.py`
- Bandit/Online learner within RL stacks: `training_orchestrator/online_learner.py`, advanced RL modules
- Components to selectively reuse:
  - Journey DB: `user_journey_database.py` (BQ `gaelp_users.*` tables)
  - Attribution models: `attribution_models.py`
  - Creative selector: `creative_selector.py`
  - Budget pacer: `budget_pacer.py`
- Components to keep lab‑only: RecSim, AuctionGym, CompetitorAgents, DeepMind/WorldModel, big HTML dashboards (`templates/gaelp_dashboard_*.html`).

## Keep vs. Park (Decision Log)

- Keep in AELP2 control loop: Orchestrator (safety/HITL/telemetry), Ads/GA4 pipelines, fidelity/readiness, shadow canary, Next.js dashboard.
- Reuse from AELP into AELP2:
  - `creative_selector.py` for bandit arms
  - `budget_pacer.py` for pacing constraints
  - `user_journey_database.py` as input for uplift/journey summaries
  - `attribution_models.py` to supply a compatible engine to `RewardAttributionWrapper`
- Park (lab‑only): MasterOrchestrator + RecSim/AuctionGym/CompetitorAgents/LLM/WorldModel; big HTML dashboards. Use RL lab for strategy what‑ifs and stress tests; export hints, not direct actions.

## Target Architecture (ASCII)

```
         Google Ads (Search, YouTube, Demand Gen/Discovery, PMax)
           Meta      TikTok      LinkedIn        (other connectors next)
               \        |            |                    /
                \       |            |                   /
                 ------------- Ads Adapters (AELP2/core/data/*) -------------
                                      |
  GA4 Export + Server Events + Offline Conversions (gclid/fbclid, user_id, hashed email)
                                      |
                                      v
                          +-----------------------------+
                          | BigQuery (facts + views)    |
                          | ads_*, ga4_*, views, logs   |
                          +-----------------+-----------+
                                            |
                          +-----------------+------------------+
                          | Identity & Journeys (gaelp_users)  |
                          | stitching + touchpoints + sessions |
                          +-----------------+------------------+
                                            |
                (weekly)                    |                    (daily/hourly)
                   v                        |                           v
       +------------------------+           |              +------------------------+
       | MMM (curves, allocs)   |<----------+--- priors ---| Bandits (TS)          |
       | (Bayesian later)       |                           | creatives/audiences   |
       +-----------+------------+                           +-----------+-----------+
                   |                                                        |
                   v                                                        v
         +---------------------+                                   +--------------------+
         | Budget Orchestrator |   HITL / Safety                  | Bandit Orchestr.   |
         | shadow canaries     |<-------------------------------> | shadow proposals   |
         +-----------+---------+                                   +---------+---------+
                     |                                                       |
                     v                                                       v
            +------------------+                                    +------------------+
            | Canary Logger    |                                    | Decisions Logger |
            +------------------+                                    +------------------+

                         RL Lab (offline simulation / what-ifs / policy hints)
                         +-----------------------------------------------+
                         |  RecSim/AuctionGym/Monte Carlo (AELP)         |
                         |  Off-policy eval → policy_hints (BQ)          |
                         +----------------------+------------------------+
                                                |
                                                v
                               Seed exploration sets, opportunity scanner,
                               and HITL reviews (no direct control)

   Value-Based Bidding Bridge: Offline conversions with predicted values → Ads APIs (EC/CAPI)
   Opportunity Scanner: Headroom/new-inventory discovery (YouTube, Demand Gen, PMax, audiences)
```

## BigQuery Data Contracts

- Existing tables/views (AELP2):
  - Tables: `training_episodes`, `safety_events`, `ab_experiments`, `fidelity_evaluations`, `canary_changes`, `canary_budgets_snapshot`, `ads_campaign_performance`, `ads_ad_performance`, `ads_conversion_action_stats`, GA4 aggregates/lagged.
  - Views: `training_episodes_daily`, `ads_campaign_daily`, `ga4_daily`, `ga4_lagged_daily`, `bidding_events_per_minute`.
- New tables (to add):
  - `mmm_curves` (channel/segment response curves + uncertainty)
  - `mmm_allocations` (proposed allocations + constraints + diagnostics)
  - `bandit_decisions` (per‑decision logging: arm, context, reward, outcome)
  - `uplift_segment_daily`, `journey_paths_daily` (causal uplift and path summaries)
  - `policy_hints` (RL lab outputs: exploration sets, budget tilts, opportunity candidates)
  - `segment_scores_daily` (propensity/uplift scores per segment and summary rollups)

## Core Jobs & Schedules

- Ingestion & Views (daily): Ads→BQ, GA4→BQ, GA4 Lag Attribution; refresh views
  - `AELP2/scripts/run_ads_ingestion.sh`, `AELP2/pipelines/ga4_to_bq.py`, `AELP2/pipelines/ga4_lagged_attribution.py`, `AELP2/pipelines/create_bq_views.py`
- Modeling & Orchestration:
  - MMM (weekly): `AELP2/pipelines/mmm_service.py` (NEW) → `mmm_curves`, `mmm_allocations`
  - Budget Orchestrator (daily; shadow): `AELP2/core/optimization/budget_orchestrator.py` (NEW) → log to `canary_changes`
  - Bandits (daily/hourly): `AELP2/core/optimization/bandit_service.py` (NEW) + `creative_selector.py` → `ab_experiments`, `bandit_decisions`
  - Uplift/Journey (weekly): `AELP2/pipelines/uplift_eval.py` (NEW), `AELP2/pipelines/journey_path_summary.py` (NEW)
- Gates & Monitoring (daily):
  - Fidelity: `AELP2/pipelines/fidelity_evaluation.py`
  - Readiness: `AELP2/scripts/check_canary_readiness.py`
- Dashboard APIs (Next.js): add `/api/bq/mmm/*`, `/api/control/allocations` (NEW)

## Multi‑Platform Onboarding & Connectors

Goal: cut setup time for non‑Google platforms and make cross‑platform bidding/ingestion turnkey.

Existing adapters (to extend): `AELP2/core/data/{google_adapter.py, meta_adapter.py, linkedin_adapter.py, tiktok_adapter.py, platform_adapter.py}`.

Deliverables:
- Platform Onboarding CLI (env‑driven, zero‑prompt where possible)
  - Scripts under `AELP2/scripts/platform_onboarding/`:
    - `init_<platform>.sh` → scaffold .env entries, verify OAuth, save credentials to `AELP2/config/.<platform>_credentials.env`
    - `create_<platform>_campaign.sh` → create paused campaign/ad set/ad group with UTMs and default budgets
    - `ingest_<platform>_to_bq.py` → backfill last N days to BQ (standard schema)
  - Admin page in Next.js (shadow‑only): shows onboarding status per platform, creds present, last ingest, and “create skeleton campaign” button (keeps campaign paused).
- Standardized BQ schema across platforms (campaign/ad performance tables with redacted names):
  - Columns: date, campaign_id, adgroup_id, ad_id, impressions, clicks, cost, conversions, revenue, ctr, cvr, avg_cpc (or equivalent), name_hash
  - Views: `<platform>_campaign_daily`, `<platform>_ad_daily` (optional)
- MMM integration: map new platform channel labels; include in `mmm_curves`/`mmm_allocations` when data available
- Bandits: plugin model to run creative/ad‑variant bandits per platform via adapters; all decisions shadow‑logged

Platforms (priority): Meta, LinkedIn, TikTok (then X/Reddit/CTV as needed)

## Implementation Plan & TODO (Phased)

### Phase A (0–6 weeks): Causal + Tactical loop in shadow
- MMM v1
  - [x] Implement `AELP2/pipelines/mmm_service.py` (bootstrap log–log + adstock) using `ads_campaign_daily` (+ optional GA4 covariates)
  - [x] Create BQ tables `mmm_curves`, `mmm_allocations` with diagnostics
  - [ ] Add `/api/bq/mmm/curves`, `/api/bq/mmm/allocations` in Next.js; render in `/exec`
- Budget Orchestrator (shadow)
  - [x] Implement `AELP2/core/optimization/budget_orchestrator.py` → compute daily direction from MMM, apply pacing caps
  - [x] Emit proposals via existing changefeed (shadow‑only) with per‑change (≤5%) and daily (≤10%) delta caps
  - [ ] Add explicit CAC guardrails, cap by historical daily spend, and log uncertainty buffers
- Bandit v1 (Creatives)
  - [x] Implement `AELP2/core/optimization/bandit_service.py` (Thompson Sampling)
  - [ ] Integrate `creative_selector.py`; reserve 10% exploration; enforce CAC caps; write to `ab_experiments` and `bandit_decisions`
  - [ ] Next.js `/creative-center`: show arms/posteriors, decisions, outcomes; HITL approvals
- Bandit Orchestrator (shadow)
  - [ ] Implement shadow controller to translate bandit decisions to executable proposals (enable/pause/split), enforce 10% exploration and CAC guardrails
  - [ ] Log proposals to changefeed; require HITL for any live mutation
- Uplift/Journey v1
  - [x] Implement `AELP2/pipelines/uplift_eval.py` to compute per‑segment uplift (exposed vs unexposed), write to `uplift_segment_daily`
  - [ ] Implement `AELP2/pipelines/journey_path_summary.py` for path counts/transition probs, write to `journey_paths_daily`
  - [ ] Feed uplift insights to MMM priors (optional first pass)
- Journeys/Segments (initial activation)
  - [ ] Populate `gaelp_users.journey_sessions` and `persistent_touchpoints` (GA4 export + server events; identity stitching with gclid/fbclid/user_id/email hash)
  - [ ] Train weekly propensity/uplift model; write `segment_scores_daily` with segment labels and scores
  - [ ] Map top segments to audiences (GA4/Ads/Meta) for routing and reporting (shadow first)
- Value‑Based Bidding (bootstrap)
  - [ ] Implement offline conversion uploads with predicted value; validate Enhanced Conversions / tROAS/value rules (shadow dry runs)
  - [ ] Prepare CAPI mappings for Meta/TikTok/LinkedIn (no‑op if creds missing)
- Attribution wrapper hardening
  - [ ] Provide compatible engine from `attribution_models.py` to `AELP2/core/intelligence/reward_attribution.py` to remove fallbacks; verify conversion‑lag integration

- Off‑the‑Shelf Integrations (bootstrap)
  - [ ] MMM: Add LightweightMMM runner (primary) and Robyn runner (validator) with Cloud Run/Prefect schedule; log credible intervals
  - [ ] Bandits: Integrate MABWiser (TS/UCB policies) under the hood for `bandit_service.py`
  - [ ] Data Quality: Add Great Expectations checks for `ads_*`, `ga4_*`, and journey tables; gate jobs on critical failures
  - [ ] Orchestration: Add Prefect flows for MMM/Bandit/Uplift/Scanner with retries and observability
  - [ ] Experimentation/HITL: Integrate GrowthBook (AB exposure logging + approvals) gating live mutations
  - [ ] Attribution sanity: Add ChannelAttribution (R) Markov/Shapley job for channel‑level cross‑check (weekly)

- Multi‑Platform Onboarding (v1)
  - [x] Add `AELP2/scripts/platform_onboarding/init_meta.sh` (env → OAuth → creds file) and campaign skeleton script
  - [x] Add `AELP2/scripts/platform_onboarding/init_linkedin.sh` and campaign skeleton script
  - [x] Add `AELP2/scripts/platform_onboarding/init_tiktok.sh` and campaign skeleton script
  - [x] Implement `AELP2/pipelines/meta_to_bq.py`, `linkedin_to_bq.py`, `tiktok_to_bq.py` with standardized schema (schema ensure)
  - [ ] Next.js admin page: Platform Onboarding status (creds present, last ingest), buttons to run skeleton creation (keeps paused) and backfills (shadow)

- Opportunity Scanner (Google first)
  - [ ] Implement headroom scans for Search (Brand vs Non-Brand) and surface expansion into YouTube, Demand Gen/Discovery, PMax (BQ queries + thresholds)
  - [ ] Create paused skeletons for selected opportunities; log to `platform_skeletons` with rationale and expected CAC/volume
  - [ ] Dashboard: “Opportunities” panel with HITL approve/deny and post-hoc tracking

### Phase B (6–12 weeks): Tighten & broaden (still shadow until metrics are green)
- MMM upgrade and segmentation
  - [ ] Adopt LightweightMMM (Bayesian) with credible intervals and seasonality; log uncertainty into `mmm_curves`; orchestrator uses lower‑bound CAC
  - [ ] Segment‑aware MMM: split prospecting vs remarketing and top uplift segments/audiences for budget tilts
  - [ ] Add geo holdouts / intent cohorts for MMM validation (optional)
- Governance and pacing
  - [ ] Tighten fidelity thresholds toward ROAS/CAC MAPE ≤ 0.5, KS ≤ 0.35 as Ads data accrues
  - [ ] Budget Orchestrator: add dayparting constraints (reuse `budget_pacer.py`), per‑campaign caps
  - [ ] Dashboard: allocation diffs, uncertainty bands, shadow vs current spend overlays
- Bandits: scale out
  - [ ] Extend bandits to audiences/keywords in high‑volume campaigns; add constrained exploration policies
  - [ ] Add segment context to bandit decisions and reporting; link to segment scores
- Value‑based bidding activation
  - [ ] Move offline conversion values from shadow to live (post HITL); verify bid response and CAC impact across platforms

- Multi‑Platform Onboarding (v2)
  - [ ] Extend ingestion to include ad‑level and audience/placement breakdowns (where APIs allow)
  - [ ] Add `<platform>_campaign_daily` and `<platform>_ad_daily` views; include in MMM scope when stable
  - [ ] Enable platform bandits (creative/ad variant) via adapters; surface in `/creative-center`
  - [ ] Google Expansion: Bring one non‑Search channel into shadow control (YouTube or Demand Gen); add bandit adapters with video metrics (VTR, view rate), map to conversions
  - [ ] PMax Exploration: ingest and evaluate; if used, constrain with negative keywords/asset groups; include in MMM

### Phase C (12–24 weeks): (Opt‑in) small live canary; scale & fortify
- [ ] With explicit approval, enable small live canary (≤5% delta, 1 campaign/change) and full rollback
- [ ] Add LTV forecasting inputs to MMM; add creative embeddings (optional)
- [ ] Keep RL lab for what‑ifs; do not put RL into production control loop

- Multi‑Platform (scale)
  - [ ] Add connectors for X/Reddit/CTV as prioritized; wire to MMM and bandits with the same schema/contracts

## Monte Carlo & Bandits in AELP (Reviewed)
- Monte Carlo (`monte_carlo_simulator.py`) is robust for parallel simulations (RL lab). We will not use it in the live control loop, but we will keep it for offline policy stress‑tests and off‑policy evaluation experiments.
- Bandits in AELP appear as part of the online learner and advanced RL stacks. For production, we implement a dedicated AELP2 bandit service (explainable TS) leveraging `creative_selector.py`, with strict safety caps, clean logging to BQ, and HITL approvals.
- RL Lab Integration Plan:
  - Keep RL in lab (no direct mutations). Output `policy_hints` (exploration sets, budget tilt ideas, new‑inventory candidates) to BQ.
  - Compare hints vs MMM/Bandit outcomes on the dashboard; allow HITL to promote hints to shadow proposals.
  - Use off‑policy evaluation and replay to gate any future live trials.

## Risks & Mitigations
- Limited Ads history (Gmail child): point MMM v1 at a historical child or broaden the window while Gmail child accrues; tighten gates later.
- Attribution wrapper fallback: provide a compatible engine to stabilize attribution‑based diagnostics.
- Fidelity initially passes only under relaxed thresholds: remain shadow‑only; tighten as data improves.

## Success Metrics
- Primary: daily new customers ↑↑, blended CAC ↓ or flat at higher volume
- Causal: marginal CAC vs MMM predictions; % spend on frontier (within uncertainty bands)
- Tactical: bandit exploration ROI; time‑to‑decision for creatives/audiences; % exploration budget utilized under CAC caps
- Governance: zero live changes without approval; shadow changefeed completeness; rollback SLAs
 - Discovery/Ramp: opportunities approved/week; time from idea→skeleton→first spend; CAC vs target per new strategy; share of spend in new channels within guardrails

## Appendix: Component Mapping

- AELP2 (keep/expand):
  - Orchestrator, BQ writer, Ads/GA4 pipelines, fidelity/readiness, shadow canary, Next.js dashboard, HITL
- AELP (reuse selectively):
  - Creative Selector, Budget Pacer, UserJourneyDatabase, Attribution Models
- AELP (lab‑only):
  - MasterOrchestrator (19–20 components), RecSim, AuctionGym, CompetitorAgents, DeepMind/WorldModel, HTML dashboards

## Component Map (Build vs Adopt)
- Data Ingestion/Adapters: Build (AELP2/core/data/*); optionally adopt RudderStack/Snowplow for event collection
- Identity/Journeys: Build stitching in BQ (gaelp_users.*); optionally adopt RudderStack ID resolution
- MMM: Adopt LightweightMMM (primary) + Robyn (validator); keep bootstrap as fallback
- Bandits: Build service; adopt MABWiser internally for algorithms
- Uplift/Propensity: Adopt CausalML now; consider EconML as data scale grows
- Orchestration: Adopt Prefect flows now; consider Cloud Composer (Airflow) later
- Data Quality: Adopt Great Expectations for checks; Evidently for ML drift (optional)
- Experimentation/HITL: Adopt GrowthBook for AB approvals/flags; wire to our canary guardrails
- Attribution sanity: Adopt ChannelAttribution (Markov/Shapley) as cross‑check; MMM remains causal allocator
- Opportunity Scanner: Build logic; adopt Google Ads Recommendations/Reach Planner APIs as inputs
- RL Lab: Keep AELP simulations; output `policy_hints` (no direct mutations)

## Licensing & Procurement (At‑a‑Glance)
- Open Source (self‑host, no signup): LightweightMMM (Py), Robyn (R), CausalML/EconML (Py), MABWiser/Vowpal Wabbit (Py/C++), Prefect OSS/Airflow, Great Expectations, ChannelAttribution (R), Evidently (OSS), Unleash/Flagsmith (OSS), PostHog (OSS), RudderStack (OSS core), Snowplow (OSS).
- Managed (optional, paid/free tiers): Prefect Cloud, GrowthBook Cloud, Unleash/Flagsmith Cloud, PostHog Cloud, RudderStack/Snowplow Cloud, Evidently Cloud, Cloud Composer (Airflow on GCP), BigQuery/Cloud Run/Cloud Build (usage‑based), Google Ads API (requires approved developer token; no API fee), GA4→BigQuery export (free for standard; BQ charges apply).
- Required accounts now: Google Ads developer token (Standard access), OAuth client/refresh tokens, GA4 API access (or BigQuery export), GCP project with BigQuery and Cloud Run/Build enabled.

Owner: Engineering/ML. Review cadence: weekly.
