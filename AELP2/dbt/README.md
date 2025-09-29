# dbt BigQuery (skeleton)

This is a placeholder dbt project plan. Models mirror views in SCHEMAS.sql.md.
- models/staging: ads__campaign_performance.sql, ga4__daily.sql, training__episodes.sql
- models/marts: kpi__ads_kpi_daily.sql, training__episodes_daily.sql, ads__campaign_daily.sql, ga4__lagged_daily.sql, ab__exposures_daily.sql, ab__results.sql
- tests: not_null, unique, relationships.

Run: configure profiles.yml locally and `dbt run`.
