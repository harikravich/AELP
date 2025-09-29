# Architecture Decisions (Living Record)

- RL posture: RL/bandits drive exploration only; no direct prod mutations. All platform changes require HITL and caps.
- LLM posture: copilot for planning/explanations/variants; never autopilot. All outputs structured with policy risk notes.
- External research: Use Perplexity API (cited web research) for idea generation and briefs; no PII; store citations alongside findings.
- Budget caps: per-change ≤ 5%; daily ≤ 10%; exploration 10–20% of daily budget.
- Halo mandatory: ramp decisions must include halo and cannibalization adjustments where available.
- Auditable control: every mutation logged to BQ; rollback endpoints exist for creative and budget changes.
- Data plane: BigQuery is the single source of truth; views via dbt; DQ via Great Expectations.
- Experimentation: GrowthBook-compatible assignment + exposures in BQ; unified results view with SRM.
- Publisher: publish paused by default; policy lint before publish; support RSA/PMax first.
- Privacy: no storage of non‑public minor data; consent‑gated; pseudonymize handles client-side.
