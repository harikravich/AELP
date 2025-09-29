# Test → Learn → Scale: Implementation Plan (Ironclad)

Objective: scale beyond $100k/day at CAC ≤ $70–$80 by exploring many use-case angles safely, learning quickly, and reallocating with confidence.

## Principles
- Deterministic control loop; LLM is copilot, not autopilot.
- HITL approvals for any platform mutation; per-change ≤5% and daily ≤10% caps; audit to BQ.
- Cheap exploration (10–20% budget) over Angle×Audience×Channel×LP×Offer cells.
- MMM headroom + halo adjust decisions before ramping.

## Phase P0 – Foundations & Audit Completion (This Week)
- Documentation (this PR):
  - System index, coverage matrix, data contracts, this plan.
- Dashboard hardening (no risky mutations):
  - Enforce `SerializedBigQuery` in all SSR pages and API routes; remove JSON dumps, fix “Invalid Date”.
  - Add Exec Summary (LLM optional) with “Show my proof” drawers; graceful off‑state if `OPENAI_API_KEY` absent.
  - Freshness badges on all pages; error cards with human‑readable text.
- A/B thin slice:
  - Add tables: `ab_assignments`, `ab_metrics_daily`; wire `/api/ab/assign` (server‑side) and extend `/api/ab/exposure`.
  - Experiments page v1: definition → exposure % → interim CAC + SRM badge.
- Creative Publisher scaffolding (DRY + paused):
  - Tables: `creative_publish_queue`, `creative_publish_log`.
  - APIs: `/api/control/creative/enqueue`, `/api/control/creative/publish`, `/api/control/creative/rollback`.
  - Publisher script: `pipelines/publish_google_creatives.py` for RSA/PMax (paused); YouTube attach if YouTube ID provided.
- Explore cells scaffolding:
  - Table: `explore_cells` (cell_key, spend, conv, cac, value, last_seen).
  - API/UI: basic list in Growth Lab; manual add; shows coverage.

## Phase P1 – Learn → Scale Loop (1–2 Weeks)
- Bandits/RL in loop:
  - Use MABWiser TS in `bandit_service`; persist posteriors to `bandit_posteriors`; snapshot policies to `rl_policy_snapshots`.
  - Explore 12–24 cells at $50–$150/day per cell; 10–15% exploration budget.
- Volume Ramp panel:
  - `/api/bq/mmm/{curves,allocations}`; daily lightweight MMM refresh; headroom compute; UI with daily caps and HITL “Request +$X/day”.
- Halo & interference v1:
  - Add `halo_experiments` + weekly GeoLift job; compute brand‑lift; add simple cannibalization score; show in ramp panel.
- Creative Center v1:
  - Winners table, thumbnails; policy lint; generate 3 variants (LLM); queue A/B via new APIs; preview endpoint reused.
- LP Studio v1:
  - Two templates; publish to % traffic via A/B assignments; GA4 funnel chart; `lp_tests` + `funnel_dropoffs` tables.
- Audience Factory (shadow):
  - `segments_to_audiences.py` outputs; UI buttons queue shadow exports; logs in users dataset.

## Phase P2 – Scale & Rigor (2–4 Weeks)
- Daily MMM per segment/geo; uncertainty bands routed to Ramp panel.
- Halo v2: geo‑rotations automation; interference by channel and segment; adjust ROI for ramp decisions.
- Audience sync live (with caps, opt‑out protections); export adapters for Google/Meta/TikTok.
- Reach planner integration into Ramp panel; schedule predictions vs delivery checks.
- Automated guardrail evaluator: blocks proposals if CAC/LTV limits or DQ fails; opens Change Risk drawer.

## Tables & APIs to Add (authoritative list)
- Tables: `ab_assignments`, `ab_metrics_daily`, `explore_cells`, `bandit_posteriors`, `rl_policy_snapshots`,
  `creative_publish_queue`, `creative_publish_log`, `creative_assets`, `creative_variants_scored`, `creative_policy_flags`,
  `lp_tests`, `lp_block_metrics`, `funnel_dropoffs`, `halo_experiments`, `halo_reads_daily`, `channel_interference_scores`.
- APIs: `/api/ab/assign`, `/api/bq/ab/results`, `/api/bq/explore/cells`, `/api/bq/rl/policy`, `/api/bq/halo`,
  `/api/control/creative/{enqueue,publish,rollback}`, `/api/control/lp/publish`.

## Guardrails & Compliance
- Always publish creatives paused; policy lint (Ads policy topics + moderation) before publish.
- Consent‑gated dynamic LP modules; pseudonymize inputs; keep in shadow until Legal okays storage.
- HITL + audit on budget changes, creative publishes, LP publishes; automatic rollback ready.

## Ownership & Ops
- Data: Pipelines team; Dashboard: FE/BE team; Modeling: DS; Ops: SRE/Marketing Ops.
- SLAs: Ads/GA4 ingest daily; MMM weekly (daily when ready); Halo weekly; DQ checks on nightly.
- Runbooks: update `docs/OPS_RUNBOOK.md` with new jobs and alerts.

