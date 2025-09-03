# GAELP Code Audit (Initial Pass)

Purpose: inventory modules, assess readiness, and define actions to converge on the target architecture.

## 1) Orchestration
- gaelp_production_orchestrator.py: Main orchestrator; needs consolidation, calibration step, consistent gates, unified status serializer.
- shadow_mode_manager.py: Shadow testing; keep but use single Orchestrator entry.
- statistical_ab_testing_framework.py: A/B infra; verify GA4/BigQuery logging of results.

Actions:
- Add Auction Calibration on start; centralize stage gates; reduce per-step logging; keep episode/BQ telemetry.

## 2) Environment & Simulator
- fortified_environment_no_hardcoding.py: Env wiring (discovery, attribution, journey DB, budget, auction). Candidate to become Env facade.
- auction_gym_integration_fixed.py: GSP mechanics; add calibration hooks; health_check.
- recsim_*: sequential user models (journey, temporal, fatigue); parameterize from GA4.
- creative_* / content_analyzer: creative features; ensure real creative metadata ingestion.
- conversion_lag_model.py: conversion delays; fit by channel/segment/creative.

Actions:
- Define Simulator API; plug components; add validation harness vs GA4/Ads distributions.

## 3) Agents & Policies
- fortified_rl_agent_no_hardcoding.py: multi-head DQN, convergence monitor; add warmup gates; stabilize early LR; separate platform bid heads later.
- bandit/allocation/creative (todo): add contextual bandits for allocation and creative selection.

Actions:
- Introduce PlatformAdapter for bid policies; keep allocation/creative as bandits; unify reward plumbing.

## 4) Data & Discovery
- discovery_engine.py: GA4 pipeline; cached data support; ensure schemas/contracts.
- segment_discovery.py: clustering; ensure consistent output schema (CVR under behavioral_metrics).
- user_journey_database.py / persistent_user_database*.py: user/journey persistence; now honors BIGQUERY_USERS_DATASET.

Actions:
- DATA_CONTRACTS.md; validate BigQuery schemas and env wiring.

## 5) Attribution & Budget
- attribution_system.py & attribution_models.py: MTA; delayed rewards; log integrity.
- budget_optimizer.py: pacing/allocations; pattern learner; tune logs (DEBUG for internal learning steps).

Actions:
- Make MTA module the single source of truth for rewards (sim/live);
- Add delayed reward tests.

## 6) Platform Connectors
- google_ads_gaelp_integration.py: Google Ads adapter; ensure minimal viable actions (bids, budgets, creatives) + guardrails.

Actions:
- Define PlatformAdapter interface; normalize action schema; plan Meta/TikTok adapters next.

## 7) Safety & HITL
- safety_framework/*: policy checks; approvals; emergency controls.
- bid_explainability_system.py: explainability (now quiet at INFO). 

Actions:
- Add content policy rules for behavioral health/minors; HITL approval queue for creative changes.

## 8) Monitoring & Observability
- regression_detector.py, convergence monitors; BQ writer added for episodes; dashboards in README_dashboard.

Actions:
- training_episodes table spec; CAC/ROAS/win-rate dashboard; alert thresholds.

## Gaps (High Priority)
- Auction Calibration routine (win-rate 10–30%); unify Simulator API; central reward/MTA module; PlatformAdapter abstraction; bandits for allocation/creative; stage gates; HITL workflow stubs.

## Next Steps
- Draft CALIBRATION_SPEC.md; design Simulator API; define PlatformAdapter; write DATA_CONTRACTS.md; propose refactor PR plan.

