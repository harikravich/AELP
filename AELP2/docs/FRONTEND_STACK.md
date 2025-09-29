Frontend Stack Spec (Agreed)

Stack
- Framework: Next.js (14+, App Router) + React + TypeScript
- UI: shadcn/ui + Tailwind CSS
- State/Data: TanStack Query (server cache), Zustand (UI state)
- Charts: Recharts or Visx; Plotly optional for advanced
- Auth: NextAuth (Google provider, Workspace-only) + RBAC middleware
- API: Next.js route handlers → BigQuery client; server actions for writes (HITL)
- Realtime: SSE or socket for episodes/safety stream (optional)

Pages
- /creative-center
  - Library: list creatives (ads_ad_performance), filters, redacted names (hashes) with metadata
  - AB Hub: start/log experiments (writes to `ab_experiments`), show lift metrics
  - Approvals: HITL queue, approve/deny; audit via `safety_events`
  - AI Variants (shadow): gated by flags; policy linting
- /training-center
  - Episodes: charts from `training_episodes_daily`; drill into `training_episodes` rows
  - Safety: event timeline; filters by severity/type
  - Calibration/Fidelity: latest calibration summary; fidelity results with MAPE/RMSE/KS
  - Subagents: `subagents_daily` activity; proposals and statuses
- /exec
  - KPIs: CAC, ROAS, spend, conversions, win_rate, impression share
  - Attribution: AOV vs LTV; lag windows; model choice
  - Budget: splits by channel/segment; rebalancer suggestions

Security & Compliance
- Never display free‑text campaign/ad/search terms without explicit opt‑in; default to hashes
- Enforce RBAC for HITL actions; admin for dangerous flags
- All writes audited to `safety_events` and `ab_experiments`

Rollout Plan
- Repo: FE lives in /apps/dashboard (Next.js project)
- CI: typecheck, lint, build; preview deploys
- Secrets: BQ and auth in environment; prod/staging environments
- Telemetry: basic usage logs; no PII

Open Items
- Component library theming and design system tokens
- Live update channel for episode/safety stream
- Admin UI for flags and quotas
