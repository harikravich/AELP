# Roadmap & Milestones (P0 → P2)

## P0 – Foundations (This Week)
- Docs: Index, Coverage, Data Contracts, Schemas, API Contracts, Exploration Plan, Guardrails, Compliance, Prompts.
- Dashboard hardening: serializer everywhere; no JSON dumps; error cards; Exec Summary + proof drawers.
- Experiments thin slice: `ab_assignments`, `ab_metrics_daily`, `/api/ab/assign`, results read.
- Creative publisher scaffolding: queue/log tables + enqueue/publish/rollback APIs (DRY + paused).
- Explore cells scaffolding: table + simple UI list.
- Research Agent (Perplexity) – spec + env + stubs
  - Add `PERPLEXITY_API_KEY` to env catalog; allow‑list domains; log citations.
  - Add tables `channel_candidates`, `research_findings`; seed one Balance query.
- LP Modules Framework (demo first)
  - Endpoints `/api/module/:slug/{start,status,result}` + registry + demo UI block
  - Tables: `lp_module_runs`, `consent_logs`, `module_results`; flags off by default
  - Ship Insight Preview (DEMO) + Scam Check (rules) on /balance‑insight‑v1; A/B vs /balance‑v2

Acceptance (Owner → Status)
- Exec/Finance/Growth/Journeys pages use serializer; no RSC/Invalid Date (FE/BE) → [x]
- Freshness badges render across pages (FE/BE) → [x]
- Experiments page shows at least one definition; `ab_assignments` and `ab_metrics_daily` exist (Platform Eng) → [x]
- Creative enqueue/publish returns 200 and writes to `creative_publish_queue` and `creative_publish_log` with status=paused_created (Growth Eng) → [x]
- Docs finalized: Index, Coverage, Data Contracts, Schemas, API Contracts, Exploration Plan, Guardrails, Compliance, Prompts (PM/DS/Eng) → [x]

## P1 – Learn → Scale (1–2 Weeks)
- Bandits: MABWiser TS; write posteriors/snapshots daily.
- Volume Ramp: MMM headroom APIs + UI with caps + HITL request.
- Halo v1: GeoLift weekly job; halo reads surfaced; interference score.
- Creative Center v1: winners + policy lint + 3 variants + queue A/B.
- LP Studio v1: two templates; % routing via assignments; GA4 funnel chart; LP test tables.
- Audience Factory (shadow): segment exports queued; logs only.
- Channels page (Candidates → Pilots → Live)
  - Read `channel_candidates`; show score/proofs; CTA “Request Pilot”.
  - Pilot flow: create 2–4 cells; ingest path defined; budget caps.
- LP Modules (live‑lite)
  - Insight Preview (public/allowed signals, live‑lite) + Scam Check v2 (tiny model)
  - Add Privacy Check (demo); legal review for partner integrations
- Research Agent (Perplexity): `/api/research/angles`, `/api/research/brief` wired; writes research tables; Growth Lab button to add to Explore Cells.

Acceptance
- 12–24 cells explored; `explore_cells` populated; `bandit_posteriors` snapshots daily (DS) → [~] (manual seed in place; nightly scheduler added)
- RL Insights page shows posteriors with uncertainty; Promote/Kill decisions logged (FE/BE) → [x]
- Volume Ramp panel shows MMM headroom and daily caps; submits HITL requests (FE/BE) → [x]
- Halo reads appear; interference scores computed; ramp drawer shows halo‑adjusted ROI (DS) → [~]
- Creative Center v1: winners list, policy lint, 3 variants queued to A/B (Growth Eng) → [~]
- LP Studio v1: two templates live; % routing via assignments; GA4 funnel chart; lp_tests rows appear (Growth Eng) → [x]
- Audience exports queued (shadow) with logs (Platform Eng) → [~]

## P2 – Scale & Rigor (2–4 Weeks)
- Daily MMM per segment; halo v2 (automation); audience sync live with caps; reach planner in ramp panel; automated guardrail evaluator.
