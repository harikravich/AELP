# Next Session Context (Concise)

Decisions
- RL stays central for exploration; AELP2 is the brain (deterministic control loop). LLMs are tools.
- External research: Use Perplexity API (with citations) behind a Research Agent; no PII outbound.
- Guardrails: per‑change ≤5%, daily ≤10%; writes blocked on prod by default; HITL approvals; audit to BQ.
- First use case: Aura Balance prioritized.

Locked Contracts (build against these)
- Schemas: see SCHEMAS.sql.md (ab_assignments/metrics, explore_cells, bandit_posteriors, rl_policy_snapshots, creative_publish_queue/log, lp_tests/funnel_dropoffs, halo_*, research_*).
- APIs: see API_CONTRACTS.md (assign/exposure/results, explore/rl, creative enqueue/publish/rollback, LP publish, halo, research).
- Env: PERPLEXITY_API_KEY (+ host), GOOGLE_ADS_*, GA4, dataset vars; flags in RISK_GUARDRAILS.md.

P0 Scope (start coding when ready)
- Serializer + error cards + freshness badges across pages.
- A/B thin slice: `/api/ab/assign`, extend exposure, metrics aggregation; Experiments v1 page.
- Creative publisher scaffolding: queue/log + paused publish for RSA/PMax.
- Explore cells scaffolding: table + Growth Lab list & manual add.

Balance Seed (to test quickly)
- 12 exploration cells in USE_CASES/AURA_BALANCE.md.
- Creative briefs & LP copy in USE_CASES/BALANCE_* docs.
- Seed SQL in SEED_DATA.sql.md to dry‑run endpoints.

What will be ready next
- Feature coverage matrix → 100% with file refs.
- Acceptance checklists linked in ROADMAP.md for each PR in PR_SEQUENCE.md.
- Research Agent spec finalized; endpoints in API_CONTRACTS.md; tables in SCHEMAS.sql.md.

Open Questions (optional to resolve now)
- CAC targets by channel for Balance (adjust if needed): Search ≤ $80; YouTube ≤ $95; PMax ≤ $85; Discovery ≤ $90.
- Sandbox dataset name distinct from prod for cookie switch.
- Ownership sign‑off on PRs 1–4 and next session’s coding kickoff.

