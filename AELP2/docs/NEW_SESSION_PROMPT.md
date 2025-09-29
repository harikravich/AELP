# Kickoff Prompt for New Session (Dashboard UX + Data Wiring)

Paste this into your coding assistant to start where we left off.

---
You are joining an in‑progress Next.js 14 dashboard refactor inside a larger AELP2 repo. Your objectives:
1) Fix data wiring so every page/API uses the selected dataset (cookie → sandbox|prod).
2) Eliminate RSC serialization errors by ensuring client components receive plain JSON; fix `[object Object]` timestamps.
3) Unify visuals with light shadcn/ui primitives (Button, Input, Skeleton, Toaster), consistent tables/cards/metric tiles, subtle gradients.
4) Add robust empty states and skeletons; stabilize Chat inline charts + Pin‑to‑Canvas.

Repo facts:
- App root: `AELP2/apps/dashboard` (Next.js 14 App Router + Tailwind)
- Global layout: `src/app/layout.tsx` (Inter font, Toaster, grid CSS)
- Dataset util: `src/lib/dataset.ts` — resolve dataset from cookie (`aelp-dataset` → 'sandbox' | 'prod')
- Timestamp utils: `src/lib/utils.ts` — use `fmtWhen()` and `toPlain()`
- Pages to fix: `src/app/exec/page.tsx`, `growth-lab/page.tsx`, `journeys/page.tsx`, `auctions-monitor/page.tsx`, `finance/page.tsx`, `ops/chat/page.tsx`
- Components: `src/components` (Card, MetricTile, ChatViz, ValueUploads, ui/*)
- APIs to audit: `src/app/api/bq/**` (replace BIGQUERY_TRAINING_DATASET with cookie‑based dataset); `api/chat/route.ts` (dynamic SQL planning)
- Canvas & pins: `/canvas` + `api/canvas/*` (pins stored in `gaelp_users.canvas_pins`)

Known issues to address:
- Auctions Monitor is empty without useful guidance when no data.
- Several `/api/bq/**` still hardcode `BIGQUERY_TRAINING_DATASET`.
- Some tables still show `[object Object]` in "When"; use `fmtWhen()`.
- Visuals inconsistent; adopt Button/Input/Skeleton/Toaster broadly; unify table styles; add empty states.

Acceptance criteria:
- Build passes: `bash AELP2/scripts/dev_dashboard_local.sh`
- No “Only plain objects…” digest errors. No `[object Object]` in any table "When" column.
- Dataset switcher toggling shows different data and freshness dates.
- Auctions Monitor shows either charts/metrics or an empty‑state message that names the dataset and required tables.
- Chat renders inline charts for typical KPI/GA4 queries and Pin‑to‑Canvas works.
- Visuals: consistent tables (compact density, subtle separators), cards/tiles unified, skeletons for loads, toasts on actions.

What to edit first:
1) Search for dataset hardcoding: `rg -n "BIGQUERY_TRAINING_DATASET as string" AELP2/apps/dashboard/src`
   - Replace with cookie‑based resolver everywhere.
2) Fix timestamp rendering & charts:
   - Replace `String(r.timestamp)…` with `fmtWhen(r.timestamp)` across pages.
   - Before passing to client charts, JSON‑sanitize series (e.g., `const seriesPlain = JSON.parse(JSON.stringify(series))`).
3) Visuals:
   - Replace ad‑hoc `btn-primary` and raw inputs with `Button`/`Input`.
   - Apply consistent table classes; add empty states; light gradient header.
4) APIs:
   - Audit `/api/bq/**` for dataset wiring; return friendly empty responses.
5) Test everything:
   - /exec, /growth-lab, /journeys, /finance, /auctions-monitor, /ops/chat, /control, /canvas.

Gotchas:
- Do not import google‑ads or other server‑only packages in client components.
- Do not re‑import react‑grid‑layout CSS per page (it belongs in layout).
- Keep mutation gates intact (AELP2_ALLOW_GOOGLE_MUTATIONS).

If something blocks a page (missing tables), return a helpful empty state instead of throwing.
---

