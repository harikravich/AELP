# Nested RL Creative Generation & Optimization System — Technical Specification (v1)

Status: Draft v1
Owner: Growth Engineering
Last Updated: 2025-09-20

## 0. Scope & Goals
- Reduce purchase CAC by 10–20% within 4–6 weeks via RL-driven creative selection/rotation and disciplined guided generation.
- Safely scale test throughput (50–200 guided variants/week) with simulator pre-screening and tight budget/test gates.
- Stage L1 (guided), L2 (selector RL), L3 (MMM bands) with guardrails; do not fully automate L1 or L3 on day one.

## 1. Architecture (Three Layers)
- L1 Creative (Guided, asynchronous)
  - Generates template-bound variants; brand safety + human approval; simulator pre-ranks.
- L2 Campaign RL (Selector/Allocator)
  - Promotes/retire creatives, rotates pacing, sets small cost caps, allocates test budget.
- L3 Portfolio (MMM bands)
  - Provides daily spend bands per campaign; L2 must remain within bands.

## 2. Creative DNA Schema
- JSON schema at `AELP2/docs/schemas/creative_dna.schema.json` defines required fields:
  - identity, placement/format, visuals, messaging, audio, brand elements, assets, meta, lineage.

## 3. Template Library & Constraints
- Library at `AELP2/docs/TEMPLATE_LIBRARY.md` with 10 base templates (e.g., PROOF_SOLUTION_15S_REELS, INSTALL_STEPS_20S_FEED…).
- Each template specifies: slots, allowed ranges (durations, overlays), brand placements, safe zones, disallowed content.

## 4. Simulator Gating Policy
- Source of truth: Phase‑2 v2 simulator (purchases/day MAPE ≈ 16%, 80% PI coverage ≈ 86%).
- Inputs per variant and matched control (same campaign/placement/target):
  - Predicted CAC median/p90; predicted purchases median/p10; control trailing medians.
- Accept to live test if ALL hold:
  - CAC_med ≤ 0.90 × CAC_control_med,
  - Purchases_med ≥ 1.10 × Purchases_control_med,
  - CAC_p90 ≤ 1.05 × CAC_control_med,
  - Purchases_p10 ≥ 0.90 × Purchases_control_med.
- Test design:
  - Budget ≤ 10% of campaign/day; duration ≥ 2 days; stop early on clear win/loss (SPRT style).
  - Minimum sample: ≥ 20 purchases before decision (unless clearly dominated).
- Promotion / retirement:
  - Promote if live CAC ≤ 0.90 × control for ≥ 2 consecutive days AND total ≥ 40 purchases.
  - Retire if live CAC ≥ 1.10 × control for 2 days OR freq7 > 2.5 with rising CAC.

## 5. L2 Selector RL Design
### 5.1 State (per creative, daily)
- purchases_1/3/7d, CAC_1/3/7d, CTR_1/3/7d, freq7, age_days, fatigue_slope
- sim_cac_med, sim_cac_p90, sim_purch_med, sim_purch_p10
- template_id, placement, spend_share, campaign_budget_share
- mmm_band (−1 below min, 0 in band, +1 above max)

### 5.2 Actions
- promote_k (set of ids), retire_m (set of ids)
- pacing_multiplier ∈ {0.8, 0.9, 1.0, 1.1, 1.2}
- cost_cap: {off, on_with_cap = 1.1× trailing CAC}
- test_budget_share ∈ {0, 0.02, 0.05, 0.10}

### 5.3 Reward (purchase‑true)
`r = −CAC + α·min(0, purchases − target) − β·max(0, freq7 − 2.0) − γ·band_violation`
- Defaults: α=0.1, β=0.2, γ=0.5 (tune via shadow eval)

### 5.4 Policy & Training
- Phase A: Contextual Thompson Sampling (CTS) with action masking; exploration via ε‑greedy on pacing.
- Phase B: PPO with safety layer enforcing constraints; offline warm‑start from CTS logs.

### 5.5 Constraints
- Δbudget/day ≤ 20% per creative and per campaign.
- Per‑creative spend ≤ 5% of campaign until promoted.
- Must keep campaign within MMM bands.

### 5.6 Logging & Explainability
- `rl_proposals` (features, predicted deltas, chosen actions, confidence) and `rl_actions_log` (applied, outcome).
- Human‑readable reasons: top 3 features contributing to each proposal.

## 6. L1 Guided Generation Workflows
### 6.1 Pipeline
1) Brief intake (objective, audience, offer, template candidates)
2) Template selection and slot filling (hooks, VO, overlays)
3) Generate N variants/brief (k=5–10) via API providers
4) Automated checks (brand safety, technical specs, captions)
5) Human approval (new templates, high novelty)
6) Registry (CreativeDNA + asset URIs; link lineage)
7) Simulator pre‑ranking; add top K (≤10%) to test queue

### 6.2 SLAs & Throughput
- Generation < 2h/brief; approval < 12h; throughput 50–200 variants/week.

### 6.3 Safety & Compliance
- Only approved template IDs permitted to launch; safe‑zone & caption legibility enforced.
- Music policy: −14 LUFS; no copyrighted tracks; VO transcript archived.

## 7. L3 MMM Budget Bands
### 7.1 Inputs
- Latest MMM curves (adstocked), uncertainty bands; target CAC or ROAS; channel constraints.

### 7.2 Outputs (per campaign, per day)
- `min_spend`, `base_spend`, `max_spend`; `target_cac`; `marginal_return`.

### 7.3 Rules
- L2 must keep total campaign spend within `[min, max]`.
- Promotions that push above `max` require human approval or next‑day band refresh.
- If realized CAC drifts > 15% above target for 2 days, bands tighten by 10% automatically.

## 8. Data Model & APIs
- Tables (logical):
  - `creatives`, `creative_variants`, `creative_assets`, `creative_metrics_daily`, `creative_sim_scores`, `rl_proposals`, `rl_actions_log`, `approvals`.
- APIs (draft JSON schemas under `AELP2/docs/schemas/`):
  - POST `/rl/v1/proposals` — L2 proposals (promote/retire/pace/cap/test_budget).
  - POST `/rl/v1/actions/approve` — apply subset of proposals (Controlled Mode).
  - GET `/rl/v1/status` — last proposals, approvals, outcomes.

## 9. Brand Safety & Governance
### 9.1 Automated Checks
- NLP keyword & claim checking; profanity/hate/violence filters.
- Vision model for unsafe imagery; PII redaction check.
- Technical: captions present & legible (contrast ≥ 4.5:1); audio loudness; safe zones; logo visibility.

### 9.2 Human Approval
- Required for: new template IDs; first 3 uses per template; high novelty (>0.7 cosine distance from library) or sensitive categories.

### 9.3 Auditability
- Store: CreativeDNA, policy report hash, approver id, timestamp in `approvals`.
- Every live test/promotion must reference approval artifact id.

## 10. Rewards, KPIs, and Decision Gates
### 10.1 Rewards
- RL reward (per §5.3). Calibration monthly; penalty weights recorded in config.

### 10.2 KPIs
- Purchase CAC (primary), Purchases/day, Test graduation rate (%), Creative fatigue (freq7), Time‑to‑decision, Share of spend in winners.

### 10.3 Decision Gates
- Promote: live CAC ≤ 0.90 × control for ≥ 2 days and ≥ 40 purchases.
- Retire: CAC ≥ 1.10 × control for 2 days OR freq7 > 2.5 with rising CAC.

## 11. Reporting & Dashboards
- See `AELP2/docs/REPORTING.md` for daily diff, leaderboard, fidelity panel, and alerts.

## 12. Rollout Plan
### Week 0–1 (Shadow)
- Wire L2 to produce proposals only; no live changes.
- Finalize templates & brand policy; enable simulator gating service.
### Week 2–3 (Controlled 10%)
- Approve subset of proposals within 10% test budget; enforce CAC gates and kill switches.
- Throughput: 50–100 variants/week.
### Week 4–6 (Scale Tests)
- If CAC improves ≥10% vs baseline, raise test budget to 20–30%; cost caps on stabilized ad sets.
- Publish DNA leaderboard and propose template mix adjustments.
### Week 7–10 (Template Autonomy)
- Allow L1 autonomy within approved templates only; tune multi‑objective weights.

## 13. Guardrails & Kill Switches
- Global: test spend ≤ 10% of campaign/day; Δbudget/day ≤ 20%; hard CAC guard per campaign.
- Rollback: if CAC > guard for 2 days or freq7>2.5 with rising CAC, stop changes; notify on-call.
- Manual override: disable proposals per campaign via `/rl/v1/actions/approve` with `disable=true`.

## 14. Success Criteria
- 4–6 weeks: ≥10% CAC reduction at equal or higher purchases/day; ≥15% test variants graduate.
- 8–12 weeks: ≥15–20% CAC reduction; simulator coverage ≥80%, MAPE ≤ 18%.

## 15. References
- Schemas: `AELP2/docs/schemas/*.json`
- APIs: `AELP2/docs/APIS.md`
- Templates: `AELP2/docs/TEMPLATE_LIBRARY.md`
