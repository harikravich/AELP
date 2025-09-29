Legacy → AELP2 Migration Plan (Initial)

Goals
- Preserve and integrate GAELP’s advanced components (RecSim bridge, AuctionGym wiring, attribution, budget, explainability, monitoring) into AELP2 without regressions.
- Keep env‑driven flags; unify telemetry (BQ), safety (HITL), fidelity gates, and dashboard.

Phases

Phase 0 — Backend Switch (compat, no code move)
- Add `AELP2_SIM_BACKEND` to AELP2 orchestrator: values `{auctiongym|enhanced|recsim}` default `auctiongym`.
- If `enhanced`, import GAELP enhanced_simulator via relative path and wire as env backend.
- Flags: `AELP2_USE_RECSIM=1` (if recsim libs present); default off.

Phase 1 — Targeted Ports (24–48h)
- Port modules with minimal edits and stable namespaces:
  - enhanced_simulator.py → AELP2/core/simulators/enhanced_simulator.py
  - bid_explainability_system.py → AELP2/core/explainability/bid_explainability.py
  - budget_optimizer.py → AELP2/core/optimization/budget_optimizer.py
  - attribution_system.py, attribution_models.py → AELP2/core/intelligence/attribution/
  - persistent_user_database.py (+ batched) → AELP2/core/persistence/
  - success/monitoring → AELP2/core/monitoring/
- Unify telemetry: all episode/safety/ab/fidelity writes via AELP2 BQ writer; add `bidding_events` writes behind flag.
- Keep orchestrator single source (AELP2); diff GAELP’s production orchestrator and merge missing features (budget optimizer hooks, explainability, success criteria).

Phase 2 — Consolidation (2–4 days)
- Remove duplicate env/orchestrator codepaths; standardize on AELP2 packages.
- Add Auctions Monitor: `bidding_events` table + minutely views; wire writer calls from both backends.
- Integrate Google Ads canary applier (allowlist/approvals) with GAELP agent if needed; keep dry‑run until gates pass.

Phase 3 — Optional Pretraining & Persona Sim
- Criteo offline pretraining harness: stage loader and example trainer.
- Persona‑based sim (AuraCampaign): optional sim target for Landing Labs.

Flags and Env
- `AELP2_SIM_BACKEND` = auctiongym|enhanced|recsim
- `AELP2_USE_RECSIM` = 0/1
- `AELP2_BIDDING_EVENTS_ENABLE` = 0/1
- Existing safety/fidelity/calibration flags remain unchanged.

Testing & Validation
- Keep a small set of legacy verify/* scripts as smoke tests.
- Acceptance: same KPI‑only fidelity gates; training stability; Auctions Monitor emits sane distributions; no hardcoded fallbacks.

Open Items
- Confirm external deps (recsim_* modules, edward2_patch, creative_integration) location; rehome into AELP2 or vendor with clear licenses.
- Review all recent docs (list in LEGACY_CATALOG.md) to capture nuanced behaviors and constraints before finalizing Phase 2.

