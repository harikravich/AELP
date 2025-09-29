#!/usr/bin/env python3
"""
Create GA4-based KPI views and an Ads↔GA4 reconciliation view in BigQuery.

Views created in <project>.<dataset>:
  - ga4_enrollments_daily  (source: GA4 export events if --event and --export-dataset provided; else from ga4_daily)
  - ads_vs_ga4_kpi_daily   (joins ads_kpi_daily with GA4 enrollments; computes ga4_cac)

Env required: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
Optional env: GA4_EXPORT_DATASET (fallback for --export-dataset)

Usage:
  python3 AELP2/scripts/ga4_reconcile_ads.py [--event EVENT_NAME] [--export-dataset DATASET]
"""
import os
import argparse
from google.cloud import bigquery


def create_or_replace_view(bq: bigquery.Client, view_id: str, sql: str) -> None:
    view = bigquery.Table(view_id)
    view.view_query = sql
    bq.delete_table(view_id, not_found_ok=True)
    bq.create_table(view)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', help='GA4 event name representing an enrollment (uses GA4 export dataset if provided)')
    ap.add_argument('--export-dataset', help='GA4 export dataset name (e.g., analytics_123456789)')
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    export_ds = args.export_dataset or os.getenv('GA4_EXPORT_DATASET')

    bq = bigquery.Client(project=project)
    base = f"{project}.{dataset}"

    # 1) GA4 enrollments daily (skip if a table/view already exists)
    enroll_view = f"{base}.ga4_enrollments_daily"
    exists = True
    try:
        bq.get_table(enroll_view)
    except Exception:
        exists = False

    if not exists and args.event and export_ds:
        export_ref = f"{project}.{export_ds}"
        event = args.event.replace('`', '')
        enroll_sql = f"""
            SELECT
              PARSE_DATE('%Y%m%d', event_date) AS date,
              COUNTIF(event_name = '{event}') AS enrollments
            FROM `{export_ref}.events_*`
            WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 180 DAY))
                                  AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
            GROUP BY date
            ORDER BY date
        """
        # Create the view when using export dataset
        create_or_replace_view(bq, enroll_view, enroll_sql)
    elif not exists:
        # Fallback: use ga4_daily (from ga4_aggregates) conversions as enrollments proxy
        enroll_sql = f"""
            SELECT date, SUM(conversions) AS enrollments
            FROM `{base}.ga4_daily`
            GROUP BY date
            ORDER BY date
        """
        create_or_replace_view(bq, enroll_view, enroll_sql)

    # 2) Ads vs GA4 reconciliation view
    recon_view = f"{base}.ads_vs_ga4_kpi_daily"
    recon_sql = f"""
        WITH a AS (
          SELECT date, cost, conversions AS ads_conversions, revenue, cac AS ads_cac, roas AS ads_roas
          FROM `{base}.ads_kpi_daily`
        ), g AS (
          SELECT date, enrollments AS ga4_enrollments
          FROM `{base}.ga4_enrollments_daily`
        )
        SELECT
          d.date,
          a.ads_conversions,
          a.cost,
          a.revenue,
          a.ads_cac,
          a.ads_roas,
          g.ga4_enrollments,
          SAFE_DIVIDE(a.cost, NULLIF(g.ga4_enrollments,0)) AS ga4_cac,
          (a.ads_conversions - g.ga4_enrollments) AS conv_delta
        FROM (SELECT DISTINCT date FROM a UNION DISTINCT SELECT date FROM g) d
        LEFT JOIN a USING(date)
        LEFT JOIN g USING(date)
        ORDER BY d.date
    """
    create_or_replace_view(bq, recon_view, recon_sql)

    # Optional: Google Paid Search only, if filtered enrollments table exists
    try:
      bq.get_table(f"{base}.ga4_enrollments_google_paid_search_daily")
      recon_view2 = f"{base}.ads_vs_ga4_google_kpi_daily"
      recon_sql2 = f"""
        WITH a AS (
          SELECT date, cost, conversions AS ads_conversions, revenue, cac AS ads_cac, roas AS ads_roas
          FROM `{base}.ads_kpi_daily`
        ), g AS (
          SELECT date, enrollments AS ga4_enrollments
          FROM `{base}.ga4_enrollments_google_paid_search_daily`
        )
        SELECT
          d.date,
          a.ads_conversions,
          a.cost,
          a.revenue,
          a.ads_cac,
          a.ads_roas,
          g.ga4_enrollments,
          SAFE_DIVIDE(a.cost, NULLIF(g.ga4_enrollments,0)) AS ga4_cac,
          (a.ads_conversions - g.ga4_enrollments) AS conv_delta
        FROM (SELECT DISTINCT date FROM a UNION DISTINCT SELECT date FROM g) d
        LEFT JOIN a USING(date)
        LEFT JOIN g USING(date)
        ORDER BY d.date
      """
      create_or_replace_view(bq, recon_view2, recon_sql2)
    except Exception:
      pass

    print(f"Created/updated: {recon_view} (enrollments source={'existing' if exists else ('export' if export_ds and args.event else 'ga4_daily')})")


if __name__ == '__main__':
    main()
