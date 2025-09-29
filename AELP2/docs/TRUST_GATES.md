# Trust Gates (Pilot Success Criteria)

Applies to canary pilot starting no earlier than 2025-09-08.

Gates (must all pass for T-0 → T-2 ramp):
- Fidelity (last 14 days): ROAS MAPE ≤ 0.50, CAC MAPE ≤ 0.50
- Stability (last 7 days): |Δ spend vs plan| ≤ 10%, no policy errors
- Safety: 0 platform policy violations; guardrails active; rollback tested

Data sources:
- `${BIGQUERY_TRAINING_DATASET}.kpi_consistency_checks` (CAC/ROAS diffs)
- `${BIGQUERY_TRAINING_DATASET}.ops_alerts` (policy/errors/spend alerts)
- `${BIGQUERY_TRAINING_DATASET}.canary_changes` (applier audit)

Ramp schedule:
- T-0: Shadow compare (no mutations) on allowlisted canary campaigns
- T-1: Live budgets only, ±5% max daily delta; ≤5% canary share of total spend
- T-2: Live budgets ±10% max daily delta; canary cumulative ≤10% per day

All live applies remain disabled unless `AELP2_ALLOW_GOOGLE_MUTATIONS=1` and HITL approval is recorded.

