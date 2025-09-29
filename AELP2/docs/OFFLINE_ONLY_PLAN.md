# Offline-Only Creative Search + Simulator Validation (v1)

Goal

- Prove the simulator can predict historical performance well enough to rank creatives and suggest budgets offline.
- Auto-generate large creative sets for Aura products, score them in-sim, and learn which patterns win — without any live account changes.

Scope

- No live mutations to Meta/Google APIs. All outputs are reports, JSON specs, and CSV leaderboards.
- Gates for “good simulator”: day-level MAPE 12–18% on purchases and CAC, 80% PI coverage ≥90%, and ad-level Precision@5 ≥70%.

Phases

1) Simulator v2.1 Calibration (offline)
- Rolling time-splits (train N → test 7) with leakage checks.
- Conformal PIs + isotonic calibration for purchase rate and CAC.
- Deliver: `AELP2/reports/sim_fidelity_onepager.md` with MAPE, coverage, drift by campaign.

2) Ad-Level Accuracy (offline)
- Metrics: Precision@K for “promote” flags, pairwise win-rate vs control, Kendall tau, Brier/AUC for a thresholded “works” label.
- Input: historical creatives + realized outcomes; sim scores from v2.1.
- Deliver: `AELP2/reports/ad_level_accuracy.json` + CSV; dashboard cards.

3) Creative Search (offline)
- Generate 200–500 variants per prioritized product via Template Library + LLM.
- Produce CreativeDNA JSON for every variant, run brand-safety checks, and store assets/specs.
- Deliver: `AELP2/outputs/creative_candidates/*.json` + thumbnails/refs.

4) Simulator Scoring + Leaderboard (offline)
- Score candidates under domain-randomized sims (CPC, fatigue, seasonality).
- Rank by expected purchases and CAC with calibrated PIs; select top 5–10 bundles.
- Deliver: `AELP2/reports/creative_leaderboard.csv/json` with reasons.

5) Macro Account Audit (offline)
- Pixel/CAPI dedupe, events & AEM, attribution windows; structure (CBO/ABO), placements (incl. Reels/Stories), cost-caps, frequency caps, creative coverage, UTMs.
- Deliver: `AELP2/reports/meta_account_audit.md` (checklist + remediation suggestions).

Compliance

- COPPA: no content that targets children <13; parent-focused framing only.
- FTC: truthful, substantiated claims; avoid unqualified “guarantees”; include material terms for any insurance/coverage mentions.
- Brand safety: profanity/violence/hate filters; caption legibility; audio loudness; logo safe zones.

Success Criteria

- Simulator: Purchases/CAC MAPE in 12–18% band; PI coverage ≥90%.
- Ad-level: Precision@5 ≥70%, pairwise win-rate ≥65%.
- Creative search: at least 5–10 distinct, policy-compliant bundles with strong simulated CAC and clear reasons.

Operating Flags

- `OFFLINE_MODE=1` (default). All publishers force “paused” artifacts only; no writes to ad accounts.

