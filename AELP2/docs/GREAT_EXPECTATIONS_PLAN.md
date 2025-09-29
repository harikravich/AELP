# Great Expectations Plan (DQ)

Suites
- ads_campaign_performance: spend/clicks/conv non-negative; date continuity; impression_share 0–1; joinable campaign_id
- ads_ad_performance: ad_id join; CTR/CVR ranges
- ga4_daily: conversions ≥ 0; date continuity; channel groups whitelist
- training_episodes: numeric ≥ 0; win_rate 0–1
- ab_metrics_daily: CAC/ROAS sane; completeness per active experiment

Execution
- Nightly in Cloud Run job; write results to `ops_alerts` (WARN/FAIL).
- Gate ramp proposals on FAIL for impacted metrics.

