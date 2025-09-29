# Dashboard Changes (2025-09-08)

Highlights
- Dataset switcher respected across Growth Lab, Journeys, Finance, Auctions Monitor; freshness API fixed to return `{ rows: [{ table_name, max_date }] }`.
- RSC serialization errors eliminated by JSON‑sanitizing chart series; timestamps rendered via `fmtWhen()` helper.
- ChatOps now supports dynamic SQL planning for KPI/GA4 questions and returns inline charts (`viz` block) with a "Pin to Canvas" flow backed by BigQuery (`gaelp_users.canvas_pins`).
- Introduced light shadcn‑style primitives (Button/Input/Skeleton/Toaster). Global Toaster wired in `layout.tsx`.
- Flow Canvas (`/canvas`): draggable/resizable tiles (metrics/charts/images, plus pinned charts from chat).

Outstanding
- Visual: unify cards/metric tiles/tables across all pages; add skeleton loaders and toasts to Control actions.
- Data wiring: audit all `/api/bq/**` routes to ensure dataset is selected via cookie (some still use `BIGQUERY_TRAINING_DATASET`).
- Empty states: standardize across pages (Exec/Growth/Journeys/Finance/Control) with concise guidance.

Acceptance Checklist
- Build succeeds: `bash AELP2/scripts/dev_dashboard_local.sh`.
- No RSC digest errors; no `[object Object]` in any table "When" column.
- Dataset switch toggles data and freshness badges.
- Chat renders inline charts for typical KPI/GA4 asks and pins to Canvas.
