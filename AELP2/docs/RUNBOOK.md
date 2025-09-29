# AELP2 Ops Runbook (Snapshot)

Quick rehydrate: see `AELP2/docs/REHYDRATE_CONTEXT.md` or run `bash AELP2/scripts/rehydrate_quickstart.sh`.

- Flows: `python3 AELP2/ops/prefect_flows.py` (prints install guidance if Prefect missing)
- Weekly MMM: flow task `aelp2-weekly-mmm` (curves, Robyn, ChannelAttribution, Journeys/Uplift/Segments, Post‑hoc, KPI checks, Reach Planner, Alerts)
- SLA: set `AELP2_FLOW_SLA_SEC` to enable SLA logging and ops_alerts on violation; optional Slack via `AELP2_SLACK_WEBHOOK_URL`.
- Canary:
  - Trust Gates: `AELP2/pipelines/trust_gates_evaluator.py`; gates in `docs/TRUST_GATES.md`.
  - Apply: `/api/control/apply-canary` (flag‑gated) and `scripts/apply_google_canary.py` (shadow default).
  - Rollback: `/api/control/canary-rollback` (writes intents), pause canary IDs.
- Monitoring: `ops_alerts` (spend spikes, SLA violations), panels in Exec dashboard.
- AB/Creative: `creative_ab_planner.py` (proposed tests), `copy_optimizer_stub.py`, `lp_ab_hooks_stub.py`.
- Data: audience expansion (`audience_expansion.py`), recs quick wins (`google_recommendations_scanner.py`), cross‑platform KPI (`cross_platform_kpi_daily.py`).
- Security: IAM audit note (`security_audit.py`), permissions checker (`permissions_check.py`), checklist (`docs/PERMISSIONS_CHECKLIST.md`).

All live mutations remain blocked unless HITL + explicit env `AELP2_ALLOW_*=1`.
