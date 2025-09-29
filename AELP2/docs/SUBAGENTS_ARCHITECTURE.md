Subagents Architecture (Parallel and MoE-ready)

Goals
- Let specialized subagents propose actions (creative, budget, targeting, calibration/drift, attribution diagnostics) in parallel to the main RL loop.
- Keep production-grade safety (gates + HITL) and observability (BQ telemetry).
- Provide a path to a Mixture-of-Experts (MoE) head while maintaining simple parallel orchestration.

Design Overview
- Orchestrator: central coordinator responsible for scheduling subagents, collecting proposals, safety/HITL validation, logging, and optional shadow execution.
- Subagents: small, single-responsibility components implementing a common interface.
- Safety: every proposed action goes through SafetyGates + PolicyChecker; HITL requested for high-risk actions.
- Telemetry: every proposal and outcome logged to BigQuery `subagent_events` (timestamped, JSON metadata).

Interfaces
- BaseSubagent
  - name(): str
  - is_enabled(env): bool (checks env flags)
  - run_step(state) -> Optional[action_dict] (non-blocking; returns None to skip)
  - health_check() -> Dict
  - config: each subagent reads AELP2_* flags

Scheduling
- Parallel scheduling conceptually; implemented as cooperative steps (run_step) invoked by the main orchestrator at a given cadence.
- Cadence controlled by env (e.g., AELP2_SUBAGENTS_CADENCE_STEPS=N). Default off when flags not set.

Safety/HITL
- validate_action_safety(action, metrics, context) from safety.hitl applied to each proposal.
- HITL is non-blocking when AELP2_HITL_NON_BLOCKING=1; warmup steps and min-step-for-approval supported by env (already implemented globally).

Telemetry
- BigQueryWriter.write_subagent_event({subagent, event_type, status, episode_id, metadata}).
- Store proposals, approvals, denials, shadow results.

MoE Path
- Option 1 (parallel subagents first): subagents act as advisors; the main policy optionally aggregates.
- Option 2 (MoE): replace the aggregator with a gating network; subagents become experts.
- Keep the interface stable so MoE can plug in later.

Flags
- AELP2_SUBAGENTS_ENABLE=1: enable subagent orchestration
- AELP2_SUBAGENTS_LIST="creative, budget, targeting, drift, attribution" (comma-separated)
- AELP2_SUBAGENTS_SHADOW=1: never perform real updates; only log proposals
- AELP2_SUBAGENTS_CADENCE_STEPS=N: invoke every N main steps (default 10)
- Per-subagent enables: AELP2_ENABLE_SUBAGENT_CREATIVE, AELP2_ENABLE_SUBAGENT_BUDGET, etc.

Security & Quotas
- Per-subagent rate limits and quotas enforced in orchestrator (env-driven)
  - Env: `AELP2_SUBAGENTS_MAX_PROPOSALS_PER_EPISODE` (default 10)
  - Budget guardrails: `AELP2_BUDGET_REBALANCER_MAX_DELTA_PCT` (default 0.1),
    `AELP2_BUDGET_REBALANCER_MAX_ABS_DELTA` (optional), pacer-awareness via
    `AELP2_BUDGET_REBALANCER_PACER_AWARE=1`, `AELP2_BUDGET_REBALANCER_PACER_CUT=0.5`
- All external calls via platform adapters run in shadow by default

Status
- Phase 1 — Shadow: Done
  - Orchestrator integrated via flags; proposals logged to `subagent_events`.
  - Implemented subagents: creative ideation (stub), budget rebalance (pacer-aware), targeting discovery (negatives), attribution diagnostics (KPI alignment), drift monitor (KS/MSE trigger).
  - Quotas/rate limits enforced via env.

Next Steps
- Phase 2 — Interplay: feed accepted proposals into off-policy eval and policy hints for learning.
