# Great Expectations (skeleton)

Planned DQ suites:
- ads_campaign_performance: non-negative metrics; date continuity; impression_share 0–1.
- ga4_daily: conversions ≥ 0; known channel groups.
- training_episodes: numeric ≥ 0; win_rate 0–1.
- ab_metrics_daily: CAC/ROAS sane; completeness per active experiments.

Nightly job: run_dq_checks.sh (already present) to execute suites.
