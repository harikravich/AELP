Great Expectations Project (AELP2)

This folder contains expectation suites for core tables. The runtime runner
`AELP2/ops/gx/run_checks.py` loads GE programmatically and can operate on
BigQuery tables or synthetic DataFrames for CI. Suites are recorded here for
auditing and iteration.

Suites:
- ads_campaign_performance.json
- ads_ad_performance.json
- ga4_aggregates.json
- gaelp_users.journey_sessions.json (existence only)
- gaelp_users.persistent_touchpoints.json (existence only)

To edit suites, update these JSON files or use GE CLI locally to regenerate.

