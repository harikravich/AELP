# Risk Guardrails & Approvals

Budget Ramp
- Per change cap: ≤ 5%
- Daily cap across changes: ≤ 10%
- Exploration share: 10–20% of daily budget

Decision Rules
- Promote variant/cell when posterior P(CAC ≤ target) ≥ 0.8 and SRM OK
- Kill/hold when P(CAC ≤ target) < 0.5 or DQ fails
- Halo adjust: reduce effective ROI if cannibalization detected; boost for proven brand‑lift

Safety Gating
- GATES_ENABLED=1 required, plus action‑specific ALLOW_* flags
- All publishes paused by default; require HITL to unpause or route traffic
- Full audit rows to BQ; rollback endpoints available

Data Quality & Policy
- GE suites on spend/clicks/conv ranges; block on critical failures
- Ads policy lint; moderation check on generated copy/assets; block if flagged

Escalations
- If CAC breach > 20% for 48h or SRM fails twice: auto‑freeze ramp; require re‑approval
- If platform policy strike: auto‑pause new publishes; review queue

