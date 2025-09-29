Great Expectations scaffolding (placeholder)

Planned suites:
- ads_campaign_performance: schema consistency, non-null date/customer_id, cost_micros ≥ 0
- ads_ad_performance: schema checks, impressions/clicks non-negative; clicks ≤ impressions
- ga4_aggregates: schema checks; sessions/users/conversions non-negative
- gaelp_users.*: basic schema and nullability checks

Integration: run suites in Prefect before MMM/Bandit jobs; fail fast on schema regressions.

