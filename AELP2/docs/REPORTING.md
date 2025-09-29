# Reporting & Dashboards (v1)

## Daily Diff (Proposals vs Predicted vs Actual)
- Fields: campaign_id, creative_id, action, spend, sim_cac_med, sim_purch_med, actual_cac, actual_purchases, decision (promote/retire), notes
- Output: `AELP2/reports/rl_daily_diff_YYYYMMDD.csv`

## Creative Leaderboard
- Columns: creative_id, template_id, age_days, freq7, purchases_7d, CAC_7d, status (Test/Active/Winner/Retired)

## Simulator Fidelity Panel
- Metrics: purchases/day MAPE, CAC/day MAPE, 80% PI coverage; latest JSONs from `AELP2/reports/*temporal_v2.json`

## Alerts
- Slack/Email when: CAC guard tripped, rollback triggered, simulator coverage < 75%

## Ad-Level Accuracy (Offline Only)
- Summary: `AELP2/reports/ad_level_accuracy.json`
- Detail: `AELP2/reports/ad_level_accuracy.csv`
- Metrics:
  - Precision@K (K=5,10) for “promote” picks
  - Pairwise win-rate vs control
  - Kendall tau rank agreement
  - Brier/AUC (if labels available)

## Creative Leaderboard (Offline)
- Generated candidates ranked by simulator score and calibrated intervals.
- Output: `AELP2/reports/creative_leaderboard.json` and per-candidate JSON under `AELP2/outputs/creative_candidates/`.
