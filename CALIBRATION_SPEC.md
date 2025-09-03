# Calibration Specification

Purpose: define how we fit simulator and policy calibration parameters from Aura GA4/Ads, and validate them.

## Sources
- GA4: events, sessions, channel/device/hour metrics, conversion events, session durations, pages/session.
- Ads (Google): impressions, clicks, CPC, conversions, cost, impression share, quality score proxies.
- Creative metadata: copy/assets/tags from creative library.

## Parameters to Fit
- Auction:
  - Win-rate vs bid curves per channel/segment/device/hour; CPC distributions; position distributions.
  - Competitor bid distributions; quality score multipliers; base query values.
- User/RecSim:
  - Session duration, revisit rates, fatigue parameters; temporal activity by hour/day; multi-touch path probabilities.
- Creative Response:
  - CTR/CVR by creative features, channel, segment, device; fatigue effects.
- Conversion Lag:
  - Delay distributions by channel/segment/creative and conversion type (trial/purchase/subscription).

## Procedure
1) Extract metrics from GA4/Ads (BQ views or API) for a rolling window (e.g., 30–90 days).
2) Fit distributions/curves per dimension (channel/segment/device/hour); store parameters versioned in BQ/GCS.
3) Validate by comparing simulated outputs vs. live distributions (KS tests on CTR/CVR/CPC/win-rate; MSE on curves).
4) Promote calibration only if validation passes thresholds; otherwise fallback to previous calibration.

## Auction Calibration (Startup Routine)
- Goal: ensure mid-bin bid actions target a 10–30% win-rate band.
- Steps:
  - Probe a small set of bid bins via AuctionModel; record win-rate/CPC/position.
  - Fit a scaling/offset mapping from agent bid bins → auction bid space.
  - Apply mapping for current session; respect budget/pacing caps.
- Guardrails: max probes N; stop if budget constraints engaged; write calibration result to logs + BQ.

## Validation Thresholds
- KS p-value ≥ 0.1 for CTR/CVR/CPC/win-rate distributions.
- Win-rate curve MSE ≤ ε (tune per platform).
- ROAS/CAC simulated vs. observed within ±X% over test windows.

## Versioning & Rollback
- Store calibration versions with timestamp, metrics, thresholds, and validation results.
- Rollback to last-good on validation failure or regression detection.

