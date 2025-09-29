# AELP2 Dashboard – Master TODO (Live Path)

Purpose: one accurate, action‑oriented list to get the Next.js dashboard live, useful, and safely connected to real data. This consolidates recent session notes (2025‑09‑08 CHANGES.md), MASTER_ARCHITECTURE_AND_TODO.md, TODO_RECONCILED.md, and scripts.

Scope: AELP2/apps/dashboard and its Cloud Run deployment; read‑only BQ APIs first, HITL control actions gated behind flags; slim container assumed unless noted.

Success Criteria (P0)
- Deployed aelp2-dashboard reachable; /api/connections/health returns ok:true with expected vars.
- Exec page shows CAC/ROAS/spend/conversions over last 7–30 days from `<project>.<dataset>.ads_kpi_daily`.
- Dataset switcher (prod/sandbox) toggles data and freshness badges consistently across pages.
- ChatOps can answer basic KPI/GA4 questions and “Pin to Canvas”, with rows written to `gaelp_users.canvas_pins`.
- No RSC serialization/digest errors; all tables render friendly timestamps.

P0 – Ship This Thin Slice (Today)
- Routing & Health
  - Ensure `/api/connections/health` shows: project, dataset mode, env dataset, gatesEnabled.
  - Add 502/500 friendly error cards for failed BQ calls across pages.
- KPI & Freshness
  - Verify or create view `ads_kpi_daily` via control endpoint (KPI lock) if missing.
  - Exec and Finance pages query by cookie‑selected dataset (no hardcoded env fallbacks).
  - Freshness API returns rows `[ { table_name, max_date } ]` and badges render.
- Dataset Switcher
  - Standardize cookie usage via `lib/dataset.ts` across all `/api/bq/**` routes (audit and fix stragglers).
  - Acceptance: toggling switch updates Exec, Finance, Journeys, Growth Lab, Auctions pages consistently.
- Canvas & ChatOps
  - Chat returns `viz` blocks for common asks (KPI trends, CAC by day) and supports “Pin to Canvas”.
  - `/canvas` renders draggable/resizable tiles with pinned charts and images; pins stored in `gaelp_users.canvas_pins`.
- UX Polish (minimum)
  - Unify tiles/cards metrics using shadcn primitives; add Skeleton loaders; toast success/failure on control actions.

P0 Controls – Read‑Only Safe Defaults (Slim image)
- Control endpoints exist but return 202 with instructions when Python pipelines aren’t bundled (slim).
- HITL flags default to off; no platform mutations.

P1 – Next 72 Hours (Shadow Learning)
- MMM surfacing
  - Implement `/api/bq/mmm/curves` and `/api/bq/mmm/allocations`; render on Exec “Portfolio” section.
  - Run `AELP2/pipelines/mmm_service.py` on sample data; show uncertainty bands.
- Bandit visibility
  - `/creative-center` page: show arms/posteriors from `bandit_decisions`, last 50 decisions, outcomes; no apply button yet.
  - Control: `/api/control/bandit-apply` returns 202 (shadow stub) in slim; logs intent row with gate reason.
- Onboarding admin (platform status)
  - `/onboarding`: surface creds presence (env), last ingest dates, “create skeleton” and “backfill 30d” buttons returning 202 in slim.

P2 – End of Week (HITL in Shadow, No Mutations)
- Controls with HITL
  - Wire HITL approve queues for bandit proposals → write to `bandit_change_approvals`; keep mutation calls disabled unless full image + flags set.
  - Budget page: show `canary_changes` (shadow) and readiness/fidelity gates status.
- Auth & RBAC
  - NextAuth (Google) with `NEXTAUTH_SECRET`; optional ALLOWED_EMAIL_DOMAIN; admin-only for control pages.
- Ops
  - Cloud Run service account has BigQuery Data Viewer + Job User roles; dashboards show permission checks.
  - Basic usage logs; no PII; error reporting for API handlers.

Audit – Open Items From 2025‑09‑08 CHANGES.md
- Visual: unify cards/metric tiles/tables across pages; skeleton loaders and toasts on Control actions.
- Data wiring: audit `/api/bq/**` routes to use cookie dataset (remove any direct `BIGQUERY_TRAINING_DATASET` reads used for selection logic).
- Empty states: standardized messages for Exec/Growth/Journeys/Finance/Control.

Detailed Task List (Checklist)
- Health & Env
  - [ ] Add self‑check endpoint to verify required envs and dataset presence (extend `/api/connections/health`).
  - [ ] Unit test for dataset resolver to cover cookie/defaults.
- Dataset Switch
  - [ ] Grep `/api/bq/**` for `BIGQUERY_TRAINING_DATASET` and refactor to `getDatasetFromCookie()`.
  - [ ] Verify Exec, Finance, Journeys, Auctions, Growth pages respect mode.
- KPI Lock & Freshness
  - [ ] Ensure `/api/control/kpi-lock` creates `ads_kpi_daily` view idempotently.
  - [ ] `/api/bq/freshness` returns `{ rows: [{ table_name, max_date }] }` for key tables.
- Canvas & ChatOps
  - [ ] Chat SQL planner: ensure safe templates and bounded parameters; handle empty results cleanly.
  - [ ] Canvas: pin/unpin APIs; BigQuery table `gaelp_users.canvas_pins` ensure.
- MMM APIs & UI (P1)
  - [ ] `/api/bq/mmm/curves` and `/allocations` query `mmm_curves`/`mmm_allocations`.
  - [ ] Exec page portfolio panel with curves and suggested allocation table.
- Bandit Readouts (P1)
  - [ ] `/creative-center` SSR: arms/posteriors from `bandit_decisions`; link to recent outcomes.
  - [ ] `/api/control/bandit-apply` returns 202 with guidance in slim.
- Onboarding (P1)
  - [ ] `/onboarding` page: surface creds presence and last ingest; buttons return 202 in slim; logs admin audit row.
- Controls & HITL (P2)
  - [ ] Approval queue UI; write audits to `safety_events` and `bandit_change_approvals`.
  - [ ] Feature flags for any apply endpoints; default off.
- Auth & RBAC (P2)
  - [ ] NextAuth config + admin middleware for control routes.
- Ops (P2)
  - [ ] Verify Cloud Run SA roles; document runbook; add `/api/connections/health` “perms ok” section.

Acceptance Tests (Smoke)
- `GET /api/connections/health` → `{ ok: true, checks: { project, dataset_selected, dataset_mode } }`.
- `GET /api/bq/kpi/daily?days=30` returns ≥ 1 row or friendly empty state.
- Exec page loads in both prod and sandbox modes; freshness badges reflect dataset switch.
- Chat “CAC last 7 days” renders chart and can be pinned; pin appears on `/canvas` and persists reload.

Runbook Snippets
- Deploy (slim):
  - `bash AELP2/scripts/deploy_dashboard.sh --project $PROJECT --region us-central1 --service aelp2-dashboard --dataset gaelp_training --users_dataset gaelp_users --ga4 properties/308028264 --slim --tail-logs`
- Create KPI view if missing (from dashboard):
  - POST `/api/control/kpi-lock`
- Health checks:
  - GET `/api/connections/health`
  - GET `/api/bq/freshness`

Risks & Mitigations
- Missing BQ tables → show clear empty states; include “how to create” hints.
- Insufficient IAM on Cloud Run SA → add BQ Data Viewer + Job User; expose in health checks.
- Ads/GA4 credentials not present in slim → control endpoints return 202 with next steps, never attempt mutations.

Suggested Next Action (Today)
- Verify dataset switcher + KPI pipeline end‑to‑end on Cloud Run:
  1) Hit `/api/connections/health` to confirm envs and dataset mode.
  2) POST `/api/control/kpi-lock` to create `ads_kpi_daily` if needed.
  3) Load `/exec` and confirm CAC/ROAS charts + freshness badges.
  4) Ask Chat: “Plot spend and conversions last 14 days” and pin to Canvas. Confirm it appears on `/canvas`.

