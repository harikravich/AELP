# dbt Project Plan (BigQuery)

Goals
- Canonicalize views in SQL with versioning and tests.
- Provide source freshness and data contracts for key tables.

Models
- staging/
  - ads__campaign_performance.sql (source: `<ds>.ads_campaign_performance`)
  - ads__ad_performance.sql
  - ga4__daily.sql
  - training__episodes.sql
- marts/
  - kpi__ads_kpi_daily.sql (requires KPI IDs)
  - training__episodes_daily.sql
  - ads__campaign_daily.sql
  - ga4__lagged_daily.sql
  - ab__exposures_daily.sql
  - ab__results.sql (join metrics daily)
  - mmm__curves.sql, mmm__allocations.sql (expose)

Tests
- Generic: not_null, unique keys, relationships (IDs across tables)
- Custom: spend ≥ 0, clicks ≥ 0, conv ≥ 0; CAC/ROAS sane ranges

CI & Schedule
- `dbt run` on PRs (marts only), nightly full build on Cloud Run job.

