#!/usr/bin/env python3
"""
Create views to reconcile internal pacing spend vs GA4 enrollments, with CAC.

  Creates in <project>.<dataset>:
  - pacing_vs_ga4_kpi_daily
  - pacing_vs_ga4_summary (28/45/90d rollups)
  - pacing_vs_ga4_monthly (month-level tie-out)

Prereqs:
  - pacing_to_bq.py has loaded <ds>.pacing_daily (date, cost[, platform])
  - ga4_enrollments_daily exists (use ga4_event_to_bq_enrollments.py if needed)

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training
  python3 AELP2/scripts/pacing_reconcile_ga4.py
"""

import os
from google.cloud import bigquery


def create_or_replace_view(bq: bigquery.Client, view_id: str, sql: str) -> None:
    v = bigquery.Table(view_id)
    v.view_query = sql
    bq.delete_table(view_id, not_found_ok=True)
    bq.create_table(v)


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not (project and dataset):
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    base = f"{project}.{dataset}"
    bq = bigquery.Client(project=project)

    daily_sql = f"""
      WITH p AS (
        SELECT DATE(date) AS date, SUM(cost) AS cost
        FROM `{base}.pacing_daily`
        GROUP BY date
      ), g AS (
        SELECT date, enrollments
        FROM `{base}.ga4_enrollments_daily`
      )
      SELECT d.date,
             p.cost,
             g.enrollments AS ga4_enrollments,
             SAFE_DIVIDE(p.cost, NULLIF(g.enrollments,0)) AS ga4_cac
      FROM (SELECT DISTINCT date FROM p UNION DISTINCT SELECT date FROM g) d
      LEFT JOIN p USING(date)
      LEFT JOIN g USING(date)
      ORDER BY d.date
    """
    create_or_replace_view(bq, f"{base}.pacing_vs_ga4_kpi_daily", daily_sql)

    summary_sql = f"""
      WITH d AS (
        SELECT * FROM `{base}.pacing_vs_ga4_kpi_daily`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY) AND CURRENT_DATE()
      )
      SELECT
        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) THEN cost END) AS cost_28,
        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) THEN ga4_enrollments END) AS enr_28,
        SAFE_DIVIDE(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) THEN cost END), NULLIF(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) THEN ga4_enrollments END),0)) AS cac_28,

        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) THEN cost END) AS cost_45,
        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) THEN ga4_enrollments END) AS enr_45,
        SAFE_DIVIDE(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) THEN cost END), NULLIF(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 45 DAY) THEN ga4_enrollments END),0)) AS cac_45,

        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) THEN cost END) AS cost_90,
        SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) THEN ga4_enrollments END) AS enr_90,
        SAFE_DIVIDE(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) THEN cost END), NULLIF(SUM(CASE WHEN date >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) THEN ga4_enrollments END),0)) AS cac_90
      FROM d
    """
    create_or_replace_view(bq, f"{base}.pacing_vs_ga4_summary", summary_sql)

    monthly_sql = f"""
      SELECT
        DATE_TRUNC(date, MONTH) AS month,
        SUM(cost) AS cost,
        SUM(ga4_enrollments) AS ga4_enrollments,
        SAFE_DIVIDE(SUM(cost), NULLIF(SUM(ga4_enrollments),0)) AS ga4_cac
      FROM `{base}.pacing_vs_ga4_kpi_daily`
      GROUP BY month
      ORDER BY month
    """
    create_or_replace_view(bq, f"{base}.pacing_vs_ga4_monthly", monthly_sql)

    monthly_aligned_sql = f"""
      SELECT
        DATE_TRUNC(date, MONTH) AS month,
        SUM(IF(ga4_enrollments IS NOT NULL, cost, NULL)) AS cost_through_ga4,
        SUM(ga4_enrollments) AS ga4_enrollments,
        SAFE_DIVIDE(SUM(IF(ga4_enrollments IS NOT NULL, cost, NULL)), NULLIF(SUM(ga4_enrollments),0)) AS ga4_cac
      FROM `{base}.pacing_vs_ga4_kpi_daily`
      GROUP BY month
      ORDER BY month
    """
    create_or_replace_view(bq, f"{base}.pacing_vs_ga4_monthly_aligned", monthly_aligned_sql)

    # Monthly pacer vs GA4 (all pacer metrics where available via GA4 builder)
    pacer_full_sql = f"""
      WITH p AS (
        SELECT DATE_TRUNC(date, MONTH) AS month,
               SUM(spend) AS spend,
               SUM(d2c_total_subscribers) AS internal_subs
        FROM `{base}.pacing_pacer_daily`
        GROUP BY month
      ), g AS (
        SELECT DATE_TRUNC(date, MONTH) AS month,
               SUM(d2c_total_subscribers) AS ga4_subs,
               SUM(mobile_subscribers) AS ga4_mobile_subs,
               SUM(d2p_starts) AS ga4_d2p_starts
        FROM `{base}.ga4_pacer_daily`
        GROUP BY month
      )
      SELECT p.month,
             p.spend,
             p.internal_subs,
             g.ga4_subs,
             g.ga4_mobile_subs,
             g.ga4_d2p_starts,
             SAFE_DIVIDE(p.spend, NULLIF(g.ga4_subs,0)) AS ga4_cac,
             SAFE_DIVIDE(p.spend, NULLIF(p.internal_subs,0)) AS internal_cac,
             SAFE_DIVIDE(p.internal_subs, NULLIF(g.ga4_subs,0)) AS subs_vs_ga4_ratio
      FROM p LEFT JOIN g USING(month)
      ORDER BY month
    """
    create_or_replace_view(bq, f"{base}.pacer_vs_ga4_monthly_full", pacer_full_sql)

    print(f"Created/updated views: {base}.pacing_vs_ga4_kpi_daily, {base}.pacing_vs_ga4_summary, {base}.pacing_vs_ga4_monthly, {base}.pacing_vs_ga4_monthly_aligned, {base}.pacer_vs_ga4_monthly_full")


if __name__ == '__main__':
    main()
