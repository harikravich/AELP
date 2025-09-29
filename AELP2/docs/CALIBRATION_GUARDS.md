# Calibration Guards & Rolling Splits (Ad-Level)

- No peeking: Train window uses only days strictly before the test window.
- Default splits: 21→7, 14→7, and 7→7 rolling-origin backtests.
- Leakage checks: ensure ad-level predictions use only train aggregates (ad/adset/campaign) and static features.
- Frozen config recorded in v2.2 report; any change requires bumping version.

Environment flags

- `AELP2_TRAIN_DAYS` (default 21)
- `AELP2_TEST_DAYS` (default 7)
- `AELP2_PRIOR_STRENGTH_K` (default 100.0)
- `AELP2_DECAY_HL_AD` (half-life days for ad-level recency; default 7)

Outputs

- `AELP2/reports/ad_level_accuracy.json|csv` (Precision@K, pairwise win-rate, tau)
- `AELP2/reports/ad_calibration_reliability.png` (reliability curve)
- `AELP2/reports/ad_calibration_v22.md` (final summary)
