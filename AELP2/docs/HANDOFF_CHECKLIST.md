# Handoff Checklist (Pilot → Aura Prod)

- Swap Ads accounts to Aura MCC/child; set PILOT_MODE=0; keep publishes PAUSED until verified.
- Confirm GA4 property/stream filters; KPI lock view exists; lagged views fresh.
- Secrets in Secret Manager; service account roles (BQ Data Viewer/Job User).
- Jobs scheduled: ads ingest, ga4 lag, dbt, DQ, ab agg, bandits_writer, halo, module_runner, reach.
- Approvals workflow understood; rollback tested.
- Legal sign-off for any LP module LIVE; privacy notices updated.
- Guardrails set (5% per change; 10% per day) and verified in evaluator.
