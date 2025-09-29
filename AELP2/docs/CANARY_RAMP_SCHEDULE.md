# Canary Ramp Schedule

Start Window: earliest 2025-09-08 (T-0) through 2025-09-10 (T-2), subject to Trust Gates passing.

- T-0 (2025-09-08): Shadow compare only on allowlisted campaigns. No mutations. Evaluate Trust Gates.
- T-1 (2025-09-09): Live budgets ±5% per day, canary spend ≤5% of total. HITL required for each apply.
- T-2 (2025-09-10): Live budgets ±10% per day, cumulative canary ≤10%. HITL + caps enforced.

Rollback: Use `/api/control/canary-rollback` (writes rollback intents) and pause canary ids.

Notes: All “apply” actions remain disabled unless `AELP2_ALLOW_GOOGLE_MUTATIONS=1` and gate checks pass.
