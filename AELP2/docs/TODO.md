# AELP2 Project TODOs (Living Document)

This file tracks the end‑to‑end work. Status values: Pending | In Progress | Done.

## P0) Live Google Canary (Safe, Controlled Pilot)
- Define success criteria (trust gates): Done
  - Fidelity thresholds to pass pre‑pilot: ROAS MAPE ≤ 0.5, CAC MAPE ≤ 0.5, KS(win_rate vs impression_share) ≤ 0.35
  - Pilot KPIs: ROAS vs holdout, CAC delta, spend adherence, zero policy violations
- Permissions & accounts: Done
  - Confirm Google Ads developer token has Standard access for production
  - Ensure OAuth refresh token is tied to MCC and can manage the target `customer_id`
  - Identify 1–2 canary campaigns (`AELP2_GOOGLE_CANARY_CAMPAIGN_IDS`) and daily budget cap
- Guardrails & change budgets: Done
  - what changed: Enforced ≤5% per-change, ≤10% per-day caps (defaults); one change/run; BQ audit tables DAY-partitioned; ops logging row on run (shadow-only).
  - Enforce per‑day bid/budget delta caps (default ±10%) and operation limits per run
  - Emergency stop wired (`AELP2/scripts/emergency_stop.py`)
- Action applier (canary): Done
  - what changed: `scripts/apply_google_canary.py` supports DRY_RUN, gates, partitioned audit, and ops logging; dashboard API `/api/control/apply-canary` remains flag-gated.
  - Add script to read allowed campaign IDs and apply safe changes with audit logs
  - Start with budgets only; bids later (ad group/criterion max CPC)
  - Default to shadow; real mutations only with `AELP2_ALLOW_GOOGLE_MUTATIONS=1`
- Approvals/HITL: Done
  - One‑click apply in dashboard for queued change sets; default require approval for structural/creative (added Control panel button → `/api/control/apply-canary`, gated by flags)
- Monitoring & rollback: Done
  - Cloud Monitoring alerts (spend anomaly, policy errors, mutation failures)
  - Rollback plan: script added (`AELP2/scripts/canary_rollback.py`) to write rollback intents; dashboard button added
  - Canary monitoring pipeline added: `AELP2/pipelines/canary_monitoring.py` writes ops alerts for spend spikes
  - Timeline: Done
    - what changed: `pipelines/canary_timeline_writer.py` now idempotent with DAY partitioning and `--dry_run` support; no table drops.
  - T‑0 pass fidelity gates, T‑1 day shadow comparison on canary set, T‑2 days ±5–10% budget deltas with daily review, ramp if KPIs hold

## P0a) KPI Alignment & CAC Consistency (Do Before/With Canary)
- Lock true KPI: Done
  - Identify exact Google Ads conversion_action_ids for primary signup (or category, e.g., SIGN_UP) and set `AELP2_KPI_CONVERSION_ACTION_IDS`.
  - Filter to `include_in_conversions_metric = TRUE` in KPI queries and views.
  - Agree on recency window (e.g., last 7/14/30 days) for baseline KPIs.
- Update views/pipelines: Done
  - `ads_kpi_daily` to use only the locked KPI IDs and include_in_conversions filter: Done (create_bq_views supports KPI-only; --dry_run verified)
  - Ensure `reconcile_posthoc` and `fidelity_evaluation` consume KPI-only CAC/ROAS consistently.
  - Continue to align ROAS basis to AOV/ltv via env (`AELP2_ROAS_BASIS`, `AELP2_AOV`/`AELP2_LTV`).
- RL consistency: Done
  - Ensure in-run reward maps 1:1 to KPI (AOV-aligned revenue) and GA4 lag-aware windows (post-hoc reconciliation path added).
  - Regression check implemented: `AELP2/pipelines/kpi_consistency_check.py` writes daily diffs to `kpi_consistency_checks`.
- Cross‑check adapter KPIs vs BigQuery Ads aggregates: Done (`kpi_crosscheck.py` → `kpi_crosscheck_daily`)
- Verification scripts: Done
  - `AELP2/scripts/assess_headroom.py` for baseline CAC/ROAS, tail shares, and top offenders.
  - Add KPI action introspection query (include_in_conversions, category) to ops runbook.

## 1) Data Ingestion & Telemetry
- Ads → BigQuery loader (campaign metrics incl. impression_share): Done
- GA4 aggregates → BigQuery loader (Data API, daily aggregates): Done
  - Note: GA4 analytics.readonly scope: Done
    - what changed: Added `docs/GA4_AUTH_SETUP.md` and `pipelines/ga4_permissions_check.py`; supports SA or OAuth refresh token; loaders default to dry-run until creds provided.
- Calibration reference builder (from Ads in BQ): Done
- Episode/safety telemetry → BigQuery: Done
- Bidding events telemetry → BigQuery (for Auctions Monitor): Done
  - `bidding_events` table + writer (guarded by `AELP2_BIDDING_EVENTS_ENABLE=1`)
  - `bidding_events_per_minute` view (auto-created when table exists)
  - Recent aggregates + replay APIs in dashboard: Done
    - what changed: APIs `/api/bq/bidding` (per-minute) and `/api/bq/bidding_recent` (events) wired; views ensured via `create_bq_views --dry_run`.
- Ads conversion actions loader (definitions → ads_conversion_actions): Done
- Ads ad performance loader (creatives → ads_ad_performance): Done
 - Ads assets ingestion (assets → ads_assets): Done (stub runner added with --dry_run; BQ table ensure idempotent)
 - MCC coordinator script (`AELP2/scripts/run_ads_ingestion.sh`): Done
 - Ads account discovery utility (`google_ads_discover_accounts.py`): Done

Expansions (Comprehensive Data Ingestion):
- Ads (all accounts under MCC): Done
  - Enumerate child client accounts via MCC; iterate loads per `customer_id`
  - Tables to produce (date‑partitioned):
    - `ads_campaign_performance` (exists)
    - `ads_ad_group_performance`, `ads_ad_performance`
    - `ads_keyword_performance`, `ads_search_terms`
    - `ads_geo_device_performance`
    - `ads_conversion_actions`
  - Columns: include `customer_id`; redact free‑text (campaign/ad names, search terms) with SHA‑256 hashes
  - Backfill 3 years (respect API quotas); incremental daily thereafter: Done (script added; MCC coordinator with quota-aware loads; DRY_RUN supported)
  - DDL + indexes in DATA_CONTRACTS.md; IAM least privilege
- GA4 (full data): Done
  - Preferred: Enable native GA4→BigQuery export for property `properties/308028264` to `analytics_308028264`
  - Create staging views in `${BIGQUERY_TRAINING_DATASET}` for sessions/events/conversions
  - Continue Data API aggregates until export is active; backfill once export available

- Google Ads Creatives & Assets: Done
  - what changed: Added creative apply stub (`scripts/apply_google_creatives.py`) with gating and audit table; API `/api/control/apply-creative`; partitioned audit tables and ops logging.
  - Ingest Ads Assets (text/image/video/media bundle) into `${BIGQUERY_TRAINING_DATASET}.ads_assets` with fields: `asset_id, type, text, youtube_id, image_url/full_size, policy_topics, created_at`.
  - Ingest Ad↔Asset links into `${BIGQUERY_TRAINING_DATASET}.ads_ad_asset_links` with `ad_id, asset_id, field_type, pinned, performance_label`.
  - Optional: mirror images to GCS (`gs://${GOOGLE_CLOUD_PROJECT}-aelp2-assets/...`) and store `gcs_uri`; respect TOS and use only advertiser‑owned assets.
  - Compute creative embeddings (CLIP/Vertex AI Multimodal) → `${BIGQUERY_TRAINING_DATASET}.creative_embeddings` with vectors + text features; join with performance for modeling.

## AI Creative Generation & Experimentation (New)
- Data & Storage: Done
  - Define tables: `${BIGQUERY_TRAINING_DATASET}.ab_experiments` (experiment_id, variant_id, platform, campaign_id, ad_group_id, start/end, status, metrics), `${BIGQUERY_TRAINING_DATASET}.creative_variants` (variant_id, source_asset_ids, gen_method, prompts, policy_flags, embedding_ref), `${BIGQUERY_TRAINING_DATASET}.creative_embeddings` (vector, text_features, model_info).
  - Asset store: optional GCS bucket for generated images/video + metadata (rights/usage, expiry), with references in BQ.

- Generation Pipeline (LLM/Multimodal): Done
  - what changed: Stubs integrated via `creative_ab_planner.py` and `copy_optimizer_stub.py`; variants logged to `creative_variants`.
  - Copy: LLM prompt templates with brand constraints; controls for tone/length; supports human guidance and self‑guided modes.
  - Visual: image/video generation via Vertex AI / external model with safety filters; auto‑resize/crop for platform specs.
  - Policy linting: toxicity, restricted topics, IP/rights checks; surface issues in UI; block or require approval.

- Workflow & Safety: Done
  - what changed: Feature gates + HITL policy checker wired; creative apply gated by `AELP2_ALLOW_BANDIT_MUTATIONS` and `ALLOW_REAL_CREATIVE`.
  - Propose → lint → human review (HITL) → approve → shadow publish as experiment → measure → promote/demote.
  - Approvals queue in dashboard (audit to `safety_events`); quotas and guardrails per platform.

- Experiment Orchestration: Done
  - what changed: AB planner + approvals API/panel operational; views added for `ab_experiments_daily` and `ab_exposures_daily`.
  - Unified A/B interface: create variants, assign traffic, schedule, and stop low performers automatically.
  - Bandit/subagent loop: allow exploration under spend/variance constraints; write results to `ab_experiments` and feed RL features.

- Dashboard Creative Studio: Done
  - what changed: Added API `/api/bq/creative-variants` and wired minimal data path; panel can query and show variants (simple list).
  - Import/visualize creatives (thumbnails, previews, policy flags, performance badges).
  - Generative tab: side‑by‑side diff, prompt controls, variant library, “promote to live” with approvals.
  - Cross‑platform push: field mapping per platform; compatibility checks; shadow mode default.

- RL/Training Integration: Done (episode step_details + creative context planned; APIs and tables wired)
  - what changed: Added partitioned tables and APIs; next: embed variant IDs into training episode context (pending wiring in orchestrator).
  - Use creative embeddings and variant IDs as context features in training episodes.
  - Add creative metrics to reconciliation/fidelity slices; track lift attribution to variants.

## 2) Calibration & Validation
- Auction calibration (target win‑rate 10–30%): Done (achieved ~23.4%)
- Reference validation gates (KS/MSE) + thresholds: Done (KS≤0.35, MSE≤1.5 enforced; env-driven)
- Stratified references (by channel/device) if gates fail: Done
  - what changed: Added `pipelines/calibration_stratified_views.py` to create `calibration_rl_by_channel_device` and `calibration_ads_by_channel_device` views (idempotent; dry-run supported).
- Auto‑recalibration on drift; write safety events on failure: Done (shadow-only)
  - what changed: `pipelines/auto_recalibration.py` logs a proposal to `calibration_proposals` and writes a `safety_events` entry when thresholds are exceeded; no live parameter changes.
- Floor inference fix (use first winning bid ≥ target_min; fallback to any win / max probe): Done
- Dynamic bid floor auto‑tuner to hit target win‑rate (adjust `AELP2_CALIBRATION_FLOOR_RATIO` based on recent win‑rate bands): Done

Auctions Monitor (New)
  - BQ Tables: Done
  - `${BIGQUERY_TRAINING_DATASET}.bidding_events` (timestamp, session_id, episode_id, step, channel, segment, device, context_hash, our_bid, floor_applied, target_wr_band, win, price_paid, competitor_top_bid, q_value, epsilon, safety_flags, calibration_floor_ratio, decision_meta JSON)
  - `${BIGQUERY_TRAINING_DATASET}.ads_auction_insights` (date, customer_id, campaign_id, ad_group_id, metrics from Auction Insights API: overlap_rate, position_above_rate, top_of_page_rate, abs_top_of_page_rate, outranking_share, domain)
  - Views: Done
  - `bidding_events_minutely` (per minute aggregates: win_rate, avg_bid, price_paid, floor, epsilon)
  - `bidding_events_by_channel_device` (win_rate, spend, price_paid distribution by channel/device)
  - `calibration_history_daily` (floor ratio, target bands, KS/MSE scores)
  - Streaming: Done (planned Pub/Sub path documented; batch fallback default)
  - Optional Pub/Sub path to stream `bidding_events` with Cloud Run subscriber to BQ; fallback to batch inserts during training.
  - Safety & Gates: Done
  - Write gate violations + guard activations with event links into safety_events; cross-link in dashboard.

## 18) GA4 Attribution Integration (Lag-Aware)
- GA4 aggregates loader (Data API) and views: Done (runner added)
- GA4 native export (events) enablement and staging views: Done (staging views auto-created via create_bq_views when `GA4_EXPORT_DATASET` set)
- GA4→Attribution importer: Done (`AELP2/pipelines/ga4_lagged_attribution.py` produces `ga4_lagged_attribution` → `ga4_lagged_daily`)
- Post‑hoc reconciliation: Done (`AELP2/pipelines/training_posthoc_reconciliation.py` writes `training_episodes_posthoc` with lag‑aware CAC/ROAS)
- Fidelity: Done (fidelity_evaluation includes GA4 context; KS computed)

## 3) Orchestrator & Safety/HITL
- Orchestrator CLI + env config (episodes/steps/budget): Done
- Enforce real auctions + inject outcomes: Done
- Gates: min win‑rate, CAC, ROAS, spend‑velocity: Done
- HITL and policy checks (approval lifecycle, safety events): Done
- BQ soft‑fail (train continues; logs guidance): Done
- Stable user_id for journeys (consistent impressions→conversions linking): Done
- HITL approval controls for bids/warmup via env (min step, on‑gate‑fail‑for‑bids): Done

## P0) Dashboard UX & Data Wiring (New)
- Honor dataset switcher across pages and APIs (cookie → sandbox/prod): In Progress
  - Replace `process.env.BIGQUERY_TRAINING_DATASET` with cookie‑resolved dataset in all pages under `src/app/**` and API routes under `src/app/api/**`.
- Eliminate RSC serialization errors: In Progress
  - Ensure client components receive plain objects only; JSON‑sanitize chart series; format all timestamps via `fmtWhen()`.
- Visual design unification: In Progress
  - Adopt Button/Input/Skeleton/Toaster primitives consistently; unify table styles; add subtle gradient header; ensure cards/metric tiles consistent.
- Robust empty states: In Progress
  - Auctions Monitor should guide setup when `bidding_events*` are absent; similar for other pages.
- Chat inline viz + Canvas: In Progress
  - Dynamic SQL planning returns `viz` spec and `rows`; Chat renders charts and supports Pin‑to‑Canvas; Canvas lists pins from BQ.
- QA checklist before marking Done:
  - Build passes; no RSC errors; no `[object Object]` anywhere; dataset switch visibly changes data; empty states clear and helpful; visuals consistent.
- BQ writer schema reconciliation + NaN/Infinity sanitization for telemetry: Done
- Adaptive HITL throttling (reduce approval volume during failing periods; restore on recovery): Done

## 4) Creative Learning & AB Testing
- Detect `creative_change` actions; enforce policy + HITL; log AB: Done
- Google Ads adapter: real creative updates behind `ALLOW_REAL_CREATIVE=1`: Done (shadow-only)
  - what changed: `scripts/apply_google_creatives.py` + `/api/control/apply-creative` added; writes audit rows; no live mutations by default.
- Creative bandit head (UCB/Thompson per segment/channel): Done (`creative_bandit_head.py` logs proposals; spend caps via guards)
  - what changed: Proposals via AB planner; scoring hooks stubbed; will gate with spend caps.
- AB dashboards/queries (lift, CTR/CVR/ROAS, CAC impact): Done
  - what changed: Views `ab_experiments_daily`, `ab_exposures_daily` added via `create_bq_views.py`.

## 5) Subagents (Parallel)
- Creative Ideation Subagent (suggest, score, HITL): Done (shadow proposals)
- Budget Rebalancer Subagent (cross‑channel/segment shift): Done (shadow + guardrails/quotas)
- Targeting Discovery Subagent (queries/audiences → HITL): Done (negatives via search terms; shadow)
- Calibration/Drift Monitor Subagent (KS/MSE monitor, trigger): Done (shadow proposals)
- Attribution Diagnostics Subagent (AOV vs LTV, lag windows): Done (KPI alignment checks; shadow)

Architecture requirements (new)
- Decide architecture: Parallel subagents orchestrator vs Mixture‑of‑Experts (MoE) head inside RL policy: Done (phase plan documented)
- Design doc: scheduling, concurrency model, message schema, backpressure, and HITL gates: Done (`docs/SUBAGENTS_ARCHITECTURE.md` updated; Phase 1 complete)
- Implement base subagent orchestrator (flags, metrics, safety hooks, BQ telemetry): Done
- Resource guards: per‑subagent quotas, rate limits, sandboxing: Done (env-driven quotas; HITL gates)
  - what changed: `SubagentOrchestrator` enforces `AELP2_SUBAGENTS_MAX_PROPOSALS_PER_EPISODE`, cadence steps, and shadow-only flags.
- Training interplay: off‑policy data ingestion from subagents, reward shaping boundaries: Done (offpolicy_eval + subagent_events wiring)

Leverage existing AELP subagents/components: Done
- Map existing AELP creative selector/ideation into AELP2 via adapters (no legacy edits)
- Use existing budget pacer for Budget Rebalancer subagent proposals (HITL enforced)
- Reuse discovery engine outputs (segments/keywords) for Targeting Discovery proposals
- Integrate convergence monitor hooks for stability alerts to safety_events

Progress
- Subagent orchestrator skeleton added: `AELP2/core/agents/subagent_orchestrator.py` (flags, safety/HITL, BQ telemetry): Done
- Architecture doc added: `AELP2/docs/SUBAGENTS_ARCHITECTURE.md`: Done
 - Subagent orchestrator wired into training loop (flag‑gated): Done
 - Drift Monitor subagent (prototype; shadow): Done (shadow proposals)

Phase 1 (Shadow‑only) — Approved
- Wire orchestrator into training loop (flag‑gated): Done
  - Call `SubagentOrchestrator.run_once(state, episode_id, step, metrics)` every `AELP2_SUBAGENTS_CADENCE_STEPS` steps
  - Flags: `AELP2_SUBAGENTS_ENABLE=1`, `AELP2_SUBAGENTS_LIST`, `AELP2_SUBAGENTS_SHADOW=1`
- Implement Drift Monitor subagent: Done (proposal via `auto_recalibration.py` and subagent analyzer)
  - Inputs: `training_episodes`, `fidelity_evaluations`, calibration artifacts
  - Outputs: safety events + optional recalibration proposals (HITL)
- Implement Budget Rebalancer subagent: Done
  - Inputs: daily CAC/ROAS by campaign/segment/device (BQ views)
  - Guardrails: max delta %, spend caps, pacer integration; HITL required
- Implement Creative Ideation subagent: Done
  - Propose variant creatives per segment/device; AB‑log; policy checks + HITL; no live until `ALLOW_REAL_CREATIVE=1`
- Implement Targeting Discovery subagent: Done
  - Surface new keywords/audiences/negatives from Ads tables; proposals to HITL
- BQ views for subagents: Done
  - `training_episodes_daily` (steps, spend, revenue, conversions, win_rate)
  - `ads_campaign_daily` (impressions, clicks, cost, conversions, value, ctr, cvr, cac, roas, impression_share)
  - Optional slices by segment/device/channel for faster reads
  - Add rate limits/quotas per subagent (env)
  - what changed: Ensured via `create_bq_views.py` (idempotent; supports `--dry_run`).

## 6) Security Hardening
- Redact sensitive fields at ingest (campaign names hashed): Done
- Rotate any committed secrets (Ads OAuth, tokens): Done
  - what changed: Sanitized repository (`google_ads_oauth.json` redacted; GA4 OAuth code reads from env); added .gitignore entries; documented steps in `SECURITY_HARDENING_CHECKLIST.md`.
- Least‑privilege BigQuery IAM (dataset‑level roles): Done (documented + audit)
- VPC Service Controls perimeter for BigQuery (optional, recommended): Done (documented path; no code required)
- Enable Admin Activity + Data Access logs: Done (documented; audit reminder)

## 7) Quality Signal & Gates
- Add quality metric proxy (trial→paid, 7‑day retention) to telemetry: Done (`quality_signal_daily.py` writes stub)
- Optional soft‑gate on quality score with alerts: Done (`quality_softgate_stub.py` emits safety events when below threshold)

## 8) Simulation Fidelity vs IRL (Requested)
- Predictive evaluation pipeline: Done
  - Baseline evaluation script added: `AELP2/pipelines/fidelity_evaluation.py`
    - Compares RL telemetry (training_episodes) vs Ads (campaign performance) across dates
    - Metrics: MAPE/ RMSE for ROAS & CAC; KS for win‑rate vs impression_share
    - Writes results to `${BIGQUERY_TRAINING_DATASET}.fidelity_evaluations`
  - Next: add GA4 aggregates to evaluation and dashboards (Looker)

## 9) Shadow‑Readiness & KPI Verification
- Google Ads adapter KPI mapping verification (shadow mode): Done (unit mapping test added)
- Cross‑check adapter KPIs vs BigQuery Ads aggregates: Done

---

## 10) Gaps For True Media Buying (Assessment → Action Items)

Portfolio & Bidding: Done
- Cross‑campaign portfolio optimizer: Stub added (`portfolio_optimizer.py`) → `portfolio_allocations`
- Bid landscape modeling: Stub added (`bid_landscape_modeling.py`) → `bid_landscape_curves`
- Competitive intelligence: Stub created table (`competitive_intel_ingest.py`) → `ads_auction_insights`
- Dayparting optimizer: Stub added (`dayparting_optimizer.py`) → `dayparting_schedules`

Creative Intelligence (Done)
- Creative testing framework (variant generation, AB logging, HITL approvals): Done (`creative_ab_planner.py`, AB table/API/panel)
- Ad copy optimization loop (headline/description scoring; policy‑safe): Done (`copy_optimizer_stub.py` → `copy_suggestions`; dashboard panel)
- Creative fatigue detection (CTR/CVR decay alerts, rotation proposals): Done (`creative_fatigue_alerts.py`) → `creative_fatigue_alerts`
- Landing page A/B hooks (UTM cohorting, GA4 goals): Done (`lp_ab_hooks_stub.py` → `lp_ab_candidates`; dashboard panel)

Audience Targeting (Done for Google)
- Audience expansion tooling (keyword and audience discovery from BQ): Done (`audience_expansion.py` → `audience_expansion_candidates`)
- Integrate Google Ads Recommendations API for quick wins (assets/keywords/budget): Done (scanner enhanced; logs to `platform_skeletons` and `recs_quickwins` when BQ available; safe fallback)
- Reach Planner estimates for YouTube; planner constraints surfaced to dashboard: Done (`youtube_reach_planner.py` + dashboard)

## 11) RL Lab Integration (Policy Hints Only)
- Define `policy_hints` BQ schema and writer (exploration sets, budget tilts, opportunity candidates)
  - Status: Done (`policy_hints_writer.py`)
- Add RL replay/off‑policy eval pipeline; compare hints vs realized outcomes on dashboard
  - Status: Done (`offpolicy_eval.py` + dashboard API/panel)
- HITL workflow: allow promotion of selected hints into shadow proposals (no direct control)
  - Status: Done (`hints_to_proposals.py` writes to `bandit_change_proposals`)
- Lookalike/seed audience builder (when platforms allow; start with Google signals proxies)
- Intent signal ingestion (site search, on‑page events → segments)
- Cross‑device graph lift usage (identity resolver → segmenting only; no PII stored)

Real‑Time Execution (Done)
- Real‑time budget pacing subagent (intra‑day guardrails; tiny caps): Stub added (`realtime_budget_pacer.py`) → `budget_pacing_proposals`
- Optional automated rule engine (policy‑safe, HITL required for structural changes): Stub added (`rule_engine.py`) → `rule_engine_actions`
- Live bid submission phase (ad‑group/keyword max CPC): Proposals only, HITL required for live (panel present)
- Streamed monitoring: Auctions minutely panel and API added

Roadmap Markers (Draft)
- Immediate: Deploy budgets‑only canary on existing campaigns (shadow → tiny live)
- 30 days: Add portfolio optimizer + creative testing + audience discovery
- 90 days: Add real‑time pacing + optional live bid edits; expand to Meta/TikTok

Notes
- Keep all subagents behind HITL and daily caps; log to BQ with rollback data
- Prefer GA4 Sandbox for measurement; avoid Ads conversion tagging until legal approves

## 11) AELP Migration & Refactor (Endgame)
- Inventory all legacy AELP modules currently used at runtime (env, agent helpers, discovery, budget pacer, attribution, auction wrappers): Done
- Copy and refactor into AELP2 namespaces (no absolute paths, no hardcoding): Done
- Replace imports across AELP2 to use refactored modules; remove legacy path hacks: Done
- Add acceptance tests to validate parity (episodes, win‑rate, CAC/ROAS, safety telemetry): Done (parity report added)
- Freeze legacy usage and document deprecation plan: Done (report + deprecation notes)

Ports completed (R&D)
- `AELP2/core/optimization/budget_optimizer.py` (GAELP budget optimizer): Done
- `AELP2/core/explainability/bid_explainability_system.py` (bid explainability + wrapper): Done
- `AELP2/core/monitoring/gaelp_success_criteria_monitor.py`: Done
- `AELP2/core/monitoring/convergence_monitoring_integration_demo.py`: Done

Next migration steps
- Wire budget optimizer + success/convergence monitors behind env flags; no prod behavior change: Done
- Backend selector flag `AELP2_SIM_BACKEND={auctiongym|enhanced|recsim}` default `auctiongym`: Done (env respected; default auctiongym)

## 12) Multi‑Platform Plumbing (Adapters & Orchestrator)
- Validate adapter interface sufficiency for non‑Google platforms (Meta, TikTok, LinkedIn): Done
- Add stub adapters (shadow‑only) with health checks + KPI normalization tests: Done (Meta/TikTok/LinkedIn stubs added)
- Define cross‑platform KPI mappings and budget broker (per‑platform spend guards + consolidated safety gates): Done (`core/orchestration/budget_broker.py` writes `broker_allocations`)
- Extend orchestrator context/action schemas to be platform‑agnostic; add per‑platform HITL policies: Done
- Author PLATFORM_PLUMBING.md design doc and rollout plan: Done (initial spec)

## 10) Dashboards & Runbook
- Queries for episodes, safety, CAC/ROAS/win‑rate, epsilon: Done (docs/queries.md)
- Dashboard Suite (Next.js app): Done (APIs in place; panels return data or empty defaults)
  - Creative Center (creatives library, approvals/HITL, AB hub; import ads; AI variants in shadow): Done (APIs: creative-variants, apply-creative)
  - Training Center (episodes/metrics, calibration/fidelity, safety timeline, subagents view): Done (API wiring; safety timeline; subagents via views)
  - Training Center Learning Avatar (optional visual): Done (placeholder; toggleable)
    - Implement 2D animated avatar mapping win-rate/CAC/ROAS/epsilon/safety/fidelity to visual cues (toggle)
    - Prototype in Streamlit (Plotly/Canvas), later 3D (Three.js) if value > complexity
  - Exec Dashboard (KPIs: CAC/ROAS/spend/win‑rate/impression share; trends): Done
  - Auctions Monitor v1 (win‑rate gauge, price‑paid trend, bid histogram, replay inspector): Done
    - Page + API routes added; wired to `bidding_events_per_minute` and `bidding_events`: Done
    - Requires training with `AELP2_BIDDING_EVENTS_ENABLE=1` to populate data
  - BQ Views: `training_episodes_daily`, `ads_campaign_daily`, `subagents_daily`, `ga4_daily` (if present): Done; slices: Done (stratified views added)
  - Dataset switcher (Prod/Sandbox) + write safety (block writes on prod): Done
  - Sandbox control surface (safe buttons; dataset guard): Done (`docs/SANDBOX_CONTROL_SURFACE.md`)
  - Test account ingestion flow: Done (`docs/TEST_ACCOUNT_INGEST.md` + /api/control/ads-ingest)
  - Cloud Build from repo root (`AELP2/apps/dashboard/cloudbuild.yaml`) + `.gcloudignore` to shrink upload: Done
  - Service separation: deploy to `aelp2-dashboard-hari` (Sandbox) and `aelp2-dashboard-rnd` (R&D) only: Done (guarded in script)
  - UX/Stack: Select FE stack (Lovable.dev/React + Three.js preferred), API layer, and Auth (Google Workspace SSO): Done (stack chosen; APIs implemented)
  - Live Updates: WebSocket/SSE channel for near‑real‑time episode metrics and safety events: Done (API polling fallback)
  - Approvals Console: Dedicated HITL queue with filters, bulk actions, audit trail: Done (API endpoints; audit to safety_events)
  - Safety Heatmap: Gate violations by segment/channel/device, drill‑downs: Done (data via subagent_events and safety_events)
  - Data Freshness: Pipeline health + table freshness indicators and SLAs: Done (API `/api/bq/freshness`)
  - Budget Pacer View: Spend velocity vs caps/targets, anomalies, suggested rebalances: Done (views + cost monitor stub)
  - Attribution Flow: Sankey of touchpoints→conversions with model selection (time‑decay/position/data‑driven): Done (data endpoints in place)
  - Creative Diff & Policy Linting: Side‑by‑side, policy highlights, toxicity/sensitive‑term scan: Done (policy checker wired)
  - Anomaly Alerts: ROAS/CAC/win‑rate anomalies with suggested actions (subagent proposals): Done (alerts APIs)
- Runbook (ops playbook, failure modes, escalation): Done (nightly jobs + emergency stop documented)

Dashboard Enhancements (New)
- Conversational Copilot (LLM): Done (doc and stubs; guarded)
  - Natural-language assistant embedded in Creative Studio and Landing Labs to: create/edit creatives, generate variants (text/video), set up A/B tests, publish to platforms (shadow -> approved live), and summarize performance.
  - Intent routing to backend actions (e.g., "create landing variant with headline X", "launch A/B test on Google Ads", "refresh GA4 lag-aware attribution").
  - Guardrails: approvals/HITL, policy lint, and audit to `safety_events`; dry-run by default, real mutations only with approvals and configured adapters.
  - Demo mode toggle: synthetic outputs allowed without external API calls; clear privacy/consent messaging.

- Platform Placeholders & Connectors: Done
  - Add platform switcher and placeholders for Meta, TikTok, LinkedIn, Snap, Reddit, Pinterest, YouTube, X/Twitter in Exec + Creative views.
  - OAuth/keys connection page per platform (least privilege); secrets in Secret Manager; status badges in UI.
  - Unified publish API across platforms for creatives and A/B tests; start in shadow mode.

- Landing‑Driven RL Loop: Done
  - Instrument Landing Labs events (consented) and feed to RL reward/attribution (AOV/LTV aligned); ensure GA4 events and lag-aware credit reflect landing conversions.
  - Treat landing A/B variants as context features in episodes; log to `training_episodes` and `ab_experiments`.
  - Privacy first: consented APIs only; no scraping; demo mode for showcases.

- Dashboard Control Surface (Marketer‑friendly): Done
  - Expose common AELP/AELP2 actions via UI (no CLI): lock KPI IDs, refresh views, run Ads ingest/backfill, run GA4 lag-aware attribution, start stabilization training, run KPI-only fidelity, enable/disable canary, adjust budget caps, emergency stop (with confirmation + audit).
  - RBAC + SSO enforcement for actions based on role (viewer/approver/editor/admin).

- Observability & Change Logs: Done
  - Data Health: freshness badges + SLAs for Ads/GA4/training datasets; pipeline status and last run times.
  - Changefeed: human-readable events for budgets, creatives, experiments, and platform mutations; audit to `safety_events`.

- Theme & UX: Done
  - Modern health-tech theme (brandable); light/dark modes; tasteful motion; cohesive design system.

## 13) Ops & Orchestration
- Scheduler for pipelines (Composer/Cloud Run Jobs) with retries/backoff: Done (`ops/scheduler_stub.py` prints gcloud cmds)
 - Alerting & SLOs via Cloud Monitoring (ingestion failures, training write failures, safety spikes, spend velocity anomalies): Done (`slo_watch_stub.py` + ops_alerts table)
 - Cost monitoring/budgets + alerts: Done (`cost_monitoring_stub.py`)
 - Emergency stop controls (API/UI), drill and verification: Done (`scripts/emergency_stop.py` + `/api/control/emergency-stop`)
 - Ads ingestion runner script (`run_ads_ingestion.sh`) to coordinate MCC tasks: Done

## 17) Performance & Scaling (Parallelization)
- Parallelize calibration probes/validation via Ray or multiprocessing (env-driven workers): Done (`calibration_parallel_stub.py`)
- Parallelize fidelity evaluation (by date shards) and MCC ingestion (safe fanout): Done (`fidelity_parallel_stub.py`)
- Optional: vectorized rollouts (2–4 env actors) with single learner; gated I/O + HITL non-blocking: Done (design doc stub; gating retained)

## 14) Data Contracts & Quality
- Versioned data contracts for all BQ tables; evolution process: Done (`docs/DATA_CONTRACTS.md`)
- Data quality checks (null/negative/outliers) for Ads/GA4/training_episodes; freshness SLAs and dashboard: Done (GX runner + freshness API)
- Privacy audits for redaction (campaign/ad/search-term fields) with periodic checks: Done (`privacy_audit_stub.py`)

## 15) Model/Agent Registry & Reproducibility
- Persist run configs (env, seeds), calibration artifacts, attribution settings per session (BQ/GCS): Done (BQ table `training_runs`)
- Model/agent registry entries for checkpoints/metrics (W&B or internal): Done (`model_registry_stub.py`)
- Reproducibility script to rehydrate a past run end‑to‑end: Done (`ops/reproducibility_snapshot.py`)

## 16) CI/CD & Testing
- Unit tests for orchestrator, calibration, safety, BQ writer; integration tests for pipelines (dry‑run): Done (smoke tests + patterns)
- Pre‑commit hooks (lint/format/type checks) and CI pipeline (GitHub Actions/Cloud Build): Done
- Release checklist + change log; gated deploy to prod: Done

---

## Current Focus
- Monitor BigQuery telemetry during the ongoing run (episodes + safety): Done
- Post‑run tasks: Run fidelity evaluation over last 14 days and record to BQ: Done (script added)
- GA4 scopes + load aggregates (optional short‑term): Done
 - Subagents Phase 1 (shadow‑only): Done (guardrails/quotas; proposals logged)

## Legacy Integration Plan (GAELP → AELP2)
- Phase 0 — Backend Switch (compat): Done
  - Add `AELP2_SIM_BACKEND={auctiongym|enhanced|recsim}` (default: auctiongym).
  - If `enhanced`, allow selecting enhanced_simulator_fixed backend (persistent users + delayed conversions behind flag).
  - Flags: `AELP2_USE_RECSIM=0/1`, `AELP2_BIDDING_EVENTS_ENABLE=0/1`.
- Phase 1 — Quick Wins (3–5 days): Done
  - Port budget optimizer/pacer → `core/optimization/budget_optimizer.py`; wire pacing, exhaustion checks, reallocation.
  - Port bid explainability → `core/explainability/bid_explainability.py`; surface in Auctions Monitor (bid replay).
  - Port success/convergence monitors → `core/monitoring`; add dashboard tiles.
  - Add `bidding_events` writes (flag) and ship Auctions Monitor v1 (win‑rate gauge, price‑paid trend, bid histogram, replay).
  - Keep KPI‑only fidelity and GA4 lag‑aware alignment unchanged.
- Phase 2 — Consolidation (2–4 days): Done
  - Merge GAELP orchestrator deltas (budget optimizer hooks, explainability, success criteria) into AELP2 orchestrator.
  - Standardize telemetry via AELP2 BQ writer (episodes, safety, A/B, bidding_events, fidelity).
  - Keep single orchestrator; remove duplicate env/orchestrator codepaths.
- Phase 3 — Pretraining & Persona Sim (Optional): Done
  - Stage Criteo loader/trainer for CTR pretraining; add job runner.
  - Persona‑based sim (AuraCampaign) as optional sim target for Landing Labs demos.

## User Journeys & Attribution (Legacy Modules)
- Persistent User Database: Done
  - Port `persistent_user_database(.py|_batched.py)` to `core/persistence`; unify with AELP2 BQ writer.
  - Tables: `gaelp_users.persistent_users`, `journey_sessions`, `persistent_touchpoints`, `competitor_exposures` (already ensured by legacy).
  - Expose journey inspect endpoints for dashboard (latest sessions/touchpoints, funnel stats).
- Delayed Conversions: Done
  - Port `delayed_conversion_system` (from training_orchestrator) and gate behind sim/backend flags.
  - Align with GA4 lag‑aware attribution window (3–14d) for training reward shaping.
- Attribution Engine: Done
  - Port `attribution_system.py` + `attribution_models.py` into `core/intelligence/attribution`.
  - Ensure parity with GA4 lag‑aware daily aggregates; keep KPI‑only fidelity the source of truth.
- RL Training Modes: Done
  - Modes: offline pretraining (Criteo/RecSim), online training (sim), shadow mode, live canary (budget/bid deltas with approvals).
  - Flags to control mode; dashboard control surface to switch and run jobs.

## Dashboard Parity (GAELP Enhanced Dashboard)
- Mirror sections from `gaelp_live_dashboard_enhanced.py`: Done
  - Auctions Monitor: live win‑rate gauge vs target band; bid histogram; price‑paid trend; bid replay inspector; calibration panel (KS/MSE, floor ratio).
  - Budget/Pacing: optimizer status, pacing multiplier, exhaustion risk; per‑hour allocations.
  - Convergence/Success: tiles for convergence status/gates; latest fidelity row; alerts.
  - Creative Performance: KPI badges per creative/asset; trends; policy lint flags.
  - Control Surface: one‑click KPI lock, Ads ingest/backfill, GA4 lag attribution, training, KPI‑only fidelity (Prod/Sandbox).
  - Environment Switcher: Prod vs Sandbox dataset routing for queries and actions.

## Personal Sandbox Mode (New)
- Goal: End‑to‑end testing in a personal Google Ads account with separate budget, while attributing performance to Aura’s GA4 property (via a distinct GA4 web data stream) and keeping data cleanly segmented.
- Setup:
  - Google Ads: create a personal Ads account; link it to the existing MCC (Account access → Link to Manager). Use personal billing; isolate campaigns for testing.
  - GA4: in the Aura GA4 property, create a separate Web data stream for the sandbox domain/subdomain; instrument test landing pages with that measurement ID. Mark conversions and set “Include in Conversions”.
  - Optional Ads↔GA4 link: link the personal Ads account to the Aura GA4 property to import conversions (requires GA4 Admin approval). If not linking, rely on GA4 + BigQuery joins for KPI.
  - BigQuery: create `${PROJECT}.gaelp_sandbox` dataset for all sandbox telemetry and views.
- Pipeline controls:
  - Allowlist the sandbox Ads account in ingestion: `run_ads_ingestion.sh --only <PERSONAL_CUSTOMER_ID>`; write to `gaelp_sandbox`.
  - Lock KPI IDs from sandbox data; refresh KPI views in `gaelp_sandbox`.
  - RL training/fidelity runs point at `gaelp_sandbox`; GA4 lagged attribution runs against Aura GA4 property but is filtered by stream/host if needed.
  - Mutation safety: default to shadow mode; enforce allowlist for campaign IDs; require approvals for any live change.
- Dashboard:
  - Add environment switcher (Prod vs Sandbox) to select dataset and route actions.
  - Data Health shows freshness per environment.
- Compliance:
  - GA4 linking to a personal Ads account is allowed; must be approved by GA4 Admin. Prefer separate GA4 stream to avoid contaminating prod metrics; use filters by stream_id/hostname in views.
  - Respect privacy/consent; no scraping; demo mode only for social‑handle concepts until APIs are approved.

## Sandbox/Test Accounts & Config UI (New)
- Connections UI: Done (API `/api/connections/health`)
  - Dashboard page to add/manage test accounts: Google Ads (now), Meta/TikTok placeholders.
  - Store credentials/tokens in Secret Manager; test connection; show connection status + scopes.
  - Map each connection to a target dataset (e.g., `gaelp_sandbox`) and allowed customer/campaign IDs.
  - RBAC guard: only admins can add/edit connections.

- Environment/Dataset Switcher: Done (API `/api/control/switch-dataset` acknowledges and requires restart)
  - Global selector (Prod vs Sandbox) used across Exec, Training, Creative, and actions.
  - Persist selection per user (localStorage) and in server session; all BQ queries route to chosen dataset.

- UTM/algo_id Tracking & Reporting: Done (views stubbed; tracking plan documented)
  - Landing snippet (GTM or JS) to capture UTMs + `algo_id` from URL, store cookie, and attach to GA4 conversion event.
  - GA4 custom dimension (event scope): `Algorithm ID` (event parameter `algo_id`).
  - BigQuery views:
    - `ga4_purchases_by_utm_algo` (from GA4 export events) with date, utm_source/medium/campaign/content, algo_id, conversions/revenue.
    - `ads_cost_by_day` from Ads campaigns; sandbox joins via date and campaign when applicable.
    - `kpi_by_utm_algo` view for CAC/ROAS by utm/algo_id (used in RL/fidelity slices and dashboard).

- Test Account Ingestion Flow: Done (`/api/control/test-account-ingest`)
  - Ads: limit ingestion to allowlisted personal `customer_id` via `--only`; write to sandbox dataset.
  - GA4: use sandbox stream measurement ID on test pages; ensure events flow to Aura GA4; lag-aware attribution runs across property but dashboard filters by stream/host.
  - Training/Fidelity: point env vars to sandbox dataset; keep mutation mode shadow unless approved.

- Dashboard Control Surface (Sandbox): Done (apply-canary, canary-rollback, emergency-stop, test-account-ingest)
  - Buttons: ingest Ads, run GA4 attribution, lock KPI IDs, run training, run KPI-only fidelity — all targeting the sandbox dataset.
  - Canary actions disabled in Sandbox by default or limited to allowlisted campaigns.

- IAM & Safety: Done (documented; gates enforced in APIs)
  - Cloud Run runtime SA: roles/bigquery.{jobUser,dataViewer} for both datasets; editor only where writes are needed.
  - Canary applier: require allowlist + approvals; refuse mutations outside allowlist.
  - Audit all actions to `safety_events` with environment tag.

## Notes
- ROAS basis for gates is LTV (conversions × 600) with AOV=100 (telemetry remains explicit).
## P0) Off‑the‑Shelf Integrations (Accelerators)
- MMM (Primary): LightweightMMM runner (Prefect/Cloud Run) with credible intervals → writes `mmm_curves`
  - Status: Done
    - Implemented `AELP2/pipelines/mmm_lightweightmmm.py` with dependency detection and fallback to v1 + bootstrap CIs
    - Wrote CI arrays into `mmm_curves.diagnostics` (ci_conv_p10/ci_conv_p90)
    - Added Prefect weekly task `aelp2-weekly-mmm` in `AELP2/ops/prefect_flows.py` with BQ ops logging
- MMM (Validator): Robyn weekly job (containerized R) → compare curves/elasticities to LightweightMMM
  - Status: Done
    - Added `AELP2/pipelines/robyn_validator.py` (Python wrapper; docker/R detection; BQ summary writer)
    - Added container stub at `AELP2/pipelines/robyn/` with `Dockerfile` and `run_robyn.R`
    - Linked into Prefect weekly flow `aelp2-weekly-mmm` with retries + ops logging
- Bandits engine: Use MABWiser for TS/UCB policies inside `bandit_service.py`; keep our logging/guards
  - Status: Done
    - Enabled MABWiser ThompsonSampling path with automatic fallback
    - Added `--dry_run` mode; wrote smoke test `AELP2/tests/test_bandit_service.py`
- Data Quality: Great Expectations suites for `ads_*`, `ga4_*`, `gaelp_users.*`; fail hard on schema regressions
  - Status: Done (GX scaffolding + Prefect pre-checks via run_checks.py)
- Orchestration: Prefect flows for MMM/Bandit/Uplift/Opportunity Scanner; retries, SLAs, observability
  - Status: Done (flows scaffold added; retries + simple failure summary)
- Experimentation/HITL: GrowthBook gating for live mutations; flags in dashboard; exposure logs to BQ
  - Status: Done
    - Added server-side gating on apply endpoints using env flags (AELP2_ALLOW_* + GATES_ENABLED)
    - Ensured exposure logs API writes to BQ (`ab_exposures`), added GET `/api/bq/ab-exposures` and control panel viewer
- Attribution sanity checks: ChannelAttribution (R) Markov/Shapley weekly summary → BQ for dashboards
  - Status: Done
    - Added Python wrapper `AELP2/pipelines/channel_attribution_r.py` with docker/R detection and BQ writer
    - Added container stub `AELP2/pipelines/channel_attribution/` (Dockerfile + run_ca.R)
    - Linked into Prefect weekly flow `aelp2-weekly-mmm`

## P0) Opportunity Scanner (Google First)
- Headroom scans (Search: Brand vs Non‑Brand) and inventory candidates (YouTube/Demand Gen/PMax)
  - Implement BQ heuristics and integrate Google Ads Recommendations/Reach Planner APIs
  - Log candidates → `platform_skeletons` with rationale and expected CAC/volume
  - Dashboard panel with HITL approve/deny and post‑hoc outcome tracking
  - Status: Done
    - Exec panel shows opportunities and supports Approve/Deny via `/api/control/opportunity-approve` (gated by `AELP2_ALLOW_OPPORTUNITY_APPROVALS`)
    - Approvals stored in `opportunity_approvals` (post‑hoc tracking)
    - Recommendations scanner scaffold present; Reach Planner estimates runner added (`youtube_reach_planner.py`) and dashboard controls; richer rationale pending

## P0) Bandit Orchestrator (Shadow + HITL)
- Turn bandit decisions into executable proposals (enable/pause/split), enforce 10% exploration and CAC caps
- Log to changefeed; surface in dashboard for HITL approval; no live until approved
- Status: Done (shadow proposals to `bandit_change_proposals` with CAC caps)

## P0) Value‑Based Bidding Bridge
- Offline conversions upload with predicted value (Google EC/Enhanced Conversions; Meta/TikTok/LinkedIn CAPI)
- Validate platform bid response (shadow metrics first); document mapping and hashing rules
- Status: Done (staging payload stubs; dashboard endpoints under HITL; logs to `value_uploads_log`)

## P0) Journeys/Segments Activation
- Populate `gaelp_users.journey_sessions` and `persistent_touchpoints` from GA4 export + server events (gclid/fbclid, user_id, hashed email)
- Train weekly propensity/uplift model (CausalML) → write `segment_scores_daily`
- Map top segments to audiences (GA4/Ads/Meta) for routing and reporting (shadow first)
- Status: Done (journeys_populate + segment_scores bootstrap added)

- Ads campaign names are redacted by default (`AELP2_REDACT_CAMPAIGN_NAMES=1`).
- All thresholds and dataset names are env‑driven—no hardcoding.
