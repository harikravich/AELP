Production Rollout Plan

Prereqs
- Next.js app deployed behind HTTPS with access to BigQuery datasets.
- Flags: `PILOT_MODE=1`, `GATES_ENABLED=1`, `AELP2_ALLOW_*_MUTATIONS=0` initially.

Steps
1) Point `VITE_API_BASE_URL` to production Next.js domain; build and publish this app.
2) Verify read-only flows with Sandbox dataset (POST /api/dataset?mode=sandbox).
3) Enable pilot actions gradually:
   - Set `AELP2_ALLOW_OPPORTUNITY_APPROVALS=1` to record approvals (no external changes).
   - Set `AELP2_ALLOW_BANDIT_MUTATIONS=1` for shadow-only bandit apply; audit logs.
   - Keep `PILOT_MODE=1` until change propagation is verified.
4) Monitor tables: `creative_publish_queue/log`, `bandit_decisions`, `ops_alerts`, `policy_enforcement`.
5) Cutover to prod mode (POST /api/dataset?mode=prod) after sign-off.
6) Disable `PILOT_MODE` last for full publishes.

Observability
- KPIs: `/api/bq/kpi/daily`, `/api/bq/freshness` for data lag.
- Approvals latency and error rates.
- API error logs from Next.js.

