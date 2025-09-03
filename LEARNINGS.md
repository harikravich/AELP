# GAELP Learnings (Living Doc)

Purpose: capture decisions, pitfalls, calibration results, and recurring issues so we don’t relearn.

## Current State (2025-09-03)
- Startup stability: early gradient spikes; mitigated by warmup LR reductions (no emergency stop during warmup).
- Auction: 0% win-rate indicates uncalibrated initial bid mapping; add Auction Calibration to target 10–30% win-rate mid-bin.
- Spend telemetry: fixed to count only auction ‘cost’; episode metrics now accurate (CAC/ROAS).
- Logging hygiene: per-decision explainability/attribution moved to DEBUG; episode summary at INFO; episodes written to BigQuery.

## Key Decisions
- Reward = conversion_value − spend with delayed credit via MTA; same module for sim and live.
- Hierarchical control: bandit (allocation/creative) + RL (bid per platform).
- Staged rollout: offline → calibrated sim → shadow/A/B → limited live → scale with gates.
- Start with Google Ads, add other platforms via PlatformAdapter.

## Calibration Notes
- Simulator parameters (CTR/CVR, CPC, win-rate, lag) calibrated from Aura GA4/Ads; validation: distribution tests (KS) against live.
- Auction Calibration routine at orchestrator start to map bid bins to target win-rate band.

## Known Gaps / TODO
- Unify Simulator API; consolidate multiple env/orchestrator paths.
- Centralize reward/attribution in one module shared by sim/live.
- Add PlatformAdapter interface and normalize actions across platforms.
- Add HITL approval workflow for creatives; content safety rules for behavioral health.

## Postmortems / Pitfalls
- Zero wins → zero learning: always calibrate bids at startup; set guardrails to avoid extended 0% win-rate windows.
- Log floods obscure signal: keep step-level logs at DEBUG; episode & gate metrics at INFO; metrics → BigQuery.

