# Sandbox Control Surface

Endpoints (POST unless noted):
- /api/control/ga4-ingest — GA4 aggregates (requires GA4_PROPERTY_ID)
- /api/control/ga4-attribution — GA4 lag-aware attribution → ga4_lagged_attribution
- /api/control/kpi-lock?ids=... — set KPI IDs (ads_kpi_daily view refresh)
- /api/control/training-run — run training (sim) (shadow)
- /api/control/apply-canary — canary budget apply (flag-gated; shadow by default)
- /api/control/canary-rollback — write rollback intents (shadow)
- /api/control/reach-planner — write YouTube reach estimates (stub)
- /api/control/ab-approve — approve_start/approve_stop AB (shadow)
- /api/control/value-upload/google — log Google EC intent (HITL)
- /api/control/value-upload/meta — log Meta CAPI intent (HITL)

Safety:
- Writes are blocked on prod dataset; actions flagged via GATES + AELP2_ALLOW_* vars.
- Canary caps enforced in scripts; dashboard actions default to shadow-only.
