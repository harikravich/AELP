Legacy GAELP Code and Docs — Catalog (Initial)

Purpose
- Comprehensive inventory of the GAELP legacy codebase to ensure no functionality is lost as we fold it into AELP2.
- Each item lists role, dependencies, maturity, and a migration recommendation.

Legend
- Maturity: Stable | Working | Experimental | Dated
- Recommendation: Port | Wrap | Keep | Archive

Code Modules (high‑value)
- enhanced_simulator.py
  - Role: Rich simulator combining AuctionGym second‑price auctions + RecSim user model + simple real‑data calibration.
  - Deps: auction_gym_integration(_fixed), recsim_auction_bridge, recsim_user_model, edward2_patch, creative_integration, dynamic_segment_integration.
  - Maturity: Working
  - Recommendation: Port → AELP2/core/simulators/enhanced_simulator.py behind `AELP2_SIM_BACKEND=enhanced`.

- fortified_environment.py / fortified_environment_no_hardcoding.py
  - Role: Gym‑style env wiring AuctionGym, creatives, discovery; no hardcoding.
  - Deps: auction_gym_integration_fixed, discovery/creative components.
  - Maturity: Working
  - Recommendation: Port essential pieces and unify with AELP2/core/env/simulator; keep one canonical env.

- gaelp_production_orchestrator.py
  - Role: Full orchestrator wiring RL, discovery, attribution, budget, explainability, safety/HITL, Google Ads, shadow, AB tests.
  - Deps: many (see file); overlaps with AELP2 production_orchestrator.
  - Maturity: Working
  - Recommendation: Diff and merge feature deltas into AELP2/core/orchestration/production_orchestrator.py; keep a flag to select backend.

- bid_explainability_system.py
  - Role: “Why this bid?” introspection (feature→bid path).
  - Maturity: Working
  - Recommendation: Port → AELP2/core/explainability/bid_explainability.py; expose in dashboard (Auctions Monitor → Bid Replay).

- budget_optimizer.py
  - Role: Pacing, reallocation, exhaustion checks; objective selection.
  - Maturity: Working
  - Recommendation: Port → AELP2/core/optimization/budget_optimizer.py; wire into orchestrator.

- attribution_system.py / attribution_models.py
  - Role: Multi‑touch attribution engine and models.
  - Maturity: Working
  - Recommendation: Port → AELP2/core/intelligence/attribution/; align with GA4 lag‑aware and KPI‑only modes.

- persistent_user_database.py (+ batched)
  - Role: Journey/touchpoint persistence to BQ; batched writer.
  - Maturity: Working
  - Recommendation: Port → AELP2/core/persistence/persistent_user_database(_batched).py; unify with AELP2 BQ writer.

- google_ads_gaelp_integration.py
  - Role: Ads interaction/agent; prod connector.
  - Maturity: Working
  - Recommendation: Keep for canary tooling; integrate into adapters with allowlists/approvals.

- success/monitoring: gaelp_success_criteria_monitor.py, convergence_monitoring_integration_demo.py
  - Role: success gates, convergence checks.
  - Maturity: Working
  - Recommendation: Port → AELP2/core/monitoring/; tie to safety_events and dashboard tiles.

- aura_campaign_simulator.py
  - Role: Persona‑based journey sim (behavioral health personas).
  - Maturity: Experimental
  - Recommendation: Keep as optional sim for Landing Labs prototyping.

- criteo_data_loader.py
  - Role: Offline CTR/log pipeline and export.
  - Maturity: Working
  - Recommendation: Stage → AELP2/pipelines/offline/criteo_data_loader.py; optional pretraining.

Docs (recent/high‑signal)
- COMPREHENSIVE_TODO_STATUS.md — Status rollup
- AGENT_PARALLEL_EXECUTION_PLAN.md — Orchestration/parallelism
- SEGMENT_DISCOVERY_INTEGRATION_SUMMARY.md — GA4 segment discovery
- BUDGET_OPTIMIZATION_INTEGRATION_SUMMARY.md — Pacer/optimizer details
- ATTRIBUTION_INTEGRATION_SUMMARY.md — MTA integration
- SHADOW_MODE_INTEGRATION_SUMMARY.md — Shadow workflows
- DASHBOARD_FIXES_SUMMARY.md — UI remedial notes
- FALLBACK_ELIMINATION_REPORT.md — “No fallbacks” policy
- PRODUCTION_READINESS_CHECKLIST/REPORT.md — Readiness criteria
- RECSIM_INTEGRATION_STATUS_FINAL.md — RecSim status
- GAELP_TRAINING_ARCHITECTURE_ASCII.md — Architecture map
- GA4_CRITEO_ARCHITECTURE.md — Offline→online alignment
- DATA_CONTRACTS.md / CALIBRATION_SPEC.md — Contracts and calibration

Gaps / External Deps to Confirm
- recsim_auction_bridge.py, recsim_user_model.py, edward2_patch.py, creative_integration.py, dynamic_segment_integration.py — Confirm locations or rehome.

Next Steps
- Complete deep read of marked modules/docs end‑to‑end; classify Stable/Working/Experimental/Dated; update this catalog with notes and codepaths.
- Use the migration plan (see LEGACY_MIGRATION_PLAN.md) to port in phases with flags and tests.

