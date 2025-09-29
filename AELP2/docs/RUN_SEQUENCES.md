# Run Sequences & Schedule

Nightly (UTC)
- 01:00 Ads MCC ingest (last 1–14d depending on table)
- 02:00 GA4 ingest; 02:30 GA4 lag attribution
- 03:00 Views refresh (dbt)
- 03:30 A/B metrics aggregation
- 04:00 DQ checks (GE); alert on FAIL

Daily
- 05:00 Bandit posteriors & policy snapshot; write to BQ

Weekly
- Sun 06:00 MMM refresh; write curves/allocations
- Mon 07:00 GeoLift; compute halo reads

On Demand
- Creative publisher; audience sync; reach planner; value uploads (gated)

