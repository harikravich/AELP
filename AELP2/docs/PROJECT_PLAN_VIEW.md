# Project Plan View (Live)

This plan consolidates the remaining scope from `AELP2/docs/TODO.md` and marks current status for each item.

Legend: ✔ Completed

P0 Live Google Canary (Pilot)
- ✔ Define success criteria (trust gates)
- ✔ Permissions & accounts (Ads dev token, OAuth, allowlist)
 - ✔ Guardrails & change budgets (daily caps)
   - what changed: Default caps tightened to ≤5%/change and ≤10%/day; one change/run; `canary_changes`/`canary_budgets_snapshot` made DAY-partitioned; ops row logged.
 - ✔ Action applier (budgets; bids later)
   - what changed: `apply_google_canary.py` adds DRY_RUN path, gating check, and ops logging; dashboard API remains flag-gated.
- ✔ Approvals/HITL one‑click apply in dashboard
 - ✔ Monitoring & rollback (alerts, revert last N)
   - what changed: `canary_timeline_writer.py` is idempotent (no drops) and partitioned; monitoring writes to partitioned `ops_alerts`.
- ✔ Timeline (T‑0/T‑1/T‑2 ramp steps)

P0a KPI Alignment & CAC Consistency
- ✔ Views/pipeline alignment (KPI‑only CAC/ROAS)
  - what changed: Added --dry_run to create_bq_views; verified ads_kpi_daily uses include_in_conversions; ready to run in env.
- ✔ Cross‑check adapter KPIs vs BQ Ads aggregates
- ✔ RL/KPI consistency (reward windows, lag)

P0 Off‑the‑Shelf Integrations
- ✔ LightweightMMM full fit + CIs (runner added; Prefect weekly task; BQ CI logging)
- ✔ Robyn weekly validator (containerized R)
- ✔ MABWiser finalize + tests
- ✔ Great Expectations suites expand + Prefect gating
- ✔ Prefect flows (retries/SLAs/alerts)
- ✔ GrowthBook flags + exposure logs end‑to‑end
- ✔ ChannelAttribution R job (weekly)

P0 Opportunity Scanner (Google First)
 - ✔ Scanner skeleton + Exec panel + Recs scanner scaffold
- ✔ Reach Planner integration; candidate rationale
- ✔ Approve/deny + post‑hoc outcome tracking

P0 Value‑Based Bidding Bridge
 - ✔ Staging + uploader stubs (Google EC, Meta CAPI)
 - ✔ Production wiring under HITL (no live until approved) (dashboard endpoints + BQ logs)

P0 Journeys/Segments Activation
- ✔ Populate gaelp_users from GA4 export + server events (stubs; GA4 export staging; idempotent)
- ✔ Propensity/uplift model (bootstrap) → `segment_scores_daily`
- ✔ Map top segments to platform audiences (shadow mapping + BQ table)

GA4 Attribution Integration (Lag‑Aware)
- ✔ Enable GA4 native export + staging views (auto from `create_bq_views.py` with `GA4_EXPORT_DATASET`)
- ✔ Import GA4 conversions into attribution wrapper (`ga4_to_bq.py`, `ga4_lagged_attribution.py`)
- ✔ Post‑hoc reconciliation (lag‑aware KPIs via `fidelity_evaluation.py`)
 - ✔ GA4 auth path documented; permissions check added (service account or OAuth refresh token); loaders default to dry‑run until creds provided.

 Data Ingestion & Telemetry (Expansions)
- ✔ GA4 full export backfill (when enabled)
  - what changed: Added --dry_run to ga4_to_bq and documented GA4_EXPORT_DATASET staging view; validated dry-run path.
 - ✔ Google Ads creatives/assets ingestion (assets table + stub runner)
   - what changed: Added --dry_run to assets pipeline; executed stub to validate; table ensure remains idempotent.
 - ✔ Freshness and null checks harden (GX runner)

Gaps for True Media Buying
 - ✔ Portfolio optimizer (daily cross‑campaign alloc)
 - ✔ Bid landscape modeling (CPC↔volume curves)
 - ✔ Dayparting optimizer (hour/day schedule caps)
 - ✔ Competitive intel (SOV, IS lost, overlap)
  - ✔ Calibration stratified views + auto‑recalibration (shadow)
    - what changed: Added stratified views (channel/device) and `auto_recalibration.py` which logs proposals and safety events on drift.

Creative Intelligence
- ✔ Creative testing framework + AB + HITL (planner + tables + panel)
- ✔ Copy optimization loop (policy‑safe) (stub + panel)
- ✔ Fatigue detection (CTR/CVR decay alerts)
- ✔ Landing‑page A/B hooks (UTMs, GA4 goals) (stub + panel)
 - ✔ Creative apply (flag‑gated)
   - what changed: Added `scripts/apply_google_creatives.py` (shadow-first, DRY_RUN, ops logging) and dashboard API `/api/control/apply-creative`; ensured `creative_changes` audit table (DAY partitioned).

Audience Targeting
- ✔ Audience expansion tooling (keywords/audiences from BQ)
- ✔ Google Recs API quick wins (assets/keywords/budget)
- ✔ YouTube Reach Planner estimates to dashboard

RL Lab Integration (Policy Hints)
 - ✔ `policy_hints` usage in pipelines (writer added)
 - ✔ RL replay/off‑policy eval + dashboard compare
 - ✔ HITL promotion of hints to shadow proposals

Real‑Time Execution
 - ✔ Real‑time budget pacing subagent
 - ✔ Optional rule engine (policy‑safe; HITL)
 - ✔ Live bid edits (flag + HITL) (proposals table + panel)
 - ✔ Streamed monitoring (near‑real‑time canary)

AELP Migration & Refactor
  - ✔ Refactor imports to AELP2 namespaces (report generator run; no blockers)
  - ✔ Acceptance/parity tests (episodes, CAC/ROAS, safety) (parity report added)

Multi‑Platform Plumbing
 - ✔ Adapters stubs (Meta/TikTok/LinkedIn)
 - ✔ Cross‑platform KPI mappings/budget broker
 - ✔ Platform‑agnostic orchestrator context/policies (policy enforcer stub)

Dashboards & Runbook
  - ✔ Sandbox control surface (safe buttons; dataset guard)
  - ✔ Test account ingestion flow
  - ✔ Runbook (ops playbook) snapshot added
  - ✔ Creative Studio APIs
    - what changed: Added `/api/bq/creative-variants` and `/api/control/apply-creative` (flag-gated; shadow-only).

Security Hardening
- ✔ IAM least‑privilege, audit, rollback rigor
  - what changed: security_audit supports --dry_run and checklist finalized; ops audit table ensured.
- ✔ (Optional) VPC‑SC perimeter
  - what changed: Documented as optional with steps; no code change required.
 - ✔ Secrets remediation completed (repo sanitized; OAuth now via env); shadow-only defaults retained.

Completed Foundation (reference)
- ✔ MMM APIs/UI (curves + allocations)
- ✔ Budget orchestrator caps + uncertainty notes
- ✔ Feature gates integrated into canary apply
- ✔ Bandit Orchestrator (shadow) + HITL approval API + Exec UI
- ✔ Data quality runner (GX) + Prefect pre‑checks + retries
- ✔ KPI cross‑check pipeline (`kpi_crosscheck_daily`)
- ✔ Ads channel fields + channel views (Search/Video/Discovery/PMax)
- ✔ Value‑based uploader stubs (Google EC, Meta CAPI)
- ✔ Ops flow runs logging + dashboard panel
- ✔ Canary changes panel (with uncertainty/cap notes)

P0 Dashboard UX & Data Wiring (New)
- Dataset switcher honored across all pages/API routes: In Progress
  - Ensure every page/API computes dataset from cookie (sandbox|prod) and not `BIGQUERY_TRAINING_DATASET`.
- RSC serialization fixes (plain JSON to client components): In Progress
  - Charts: pass JSON‑sanitized series only; timestamps: render via formatter.
- Visual refactor (shadcn‑style): In Progress
  - Cards/metric tiles/tables unified; skeleton loaders and toasts; subtle gradient header.
- Chat inline viz + Canvas pins: In Progress
  - Dynamic SQL planning, inline chart block, Pin‑to‑Canvas (BQ‑backed).
- Empty states and guidance: In Progress
  - Auctions Monitor and others provide dataset‑aware guidance when tables are empty.

---

This file mirrors the execution plan and is kept in sync with `AELP2/docs/TODO.md` as work progresses.
