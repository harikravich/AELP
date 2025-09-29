#!/usr/bin/env python3
"""
Baseline Report (last 28 days) from BigQuery for Google Ads and GA4.

Outputs consolidated KPIs:
- Impression share (p50), impressions, clicks, conversions, CTR, CVR, CAC, ROAS (Ads)
- GA4 sessions and conversions (if available)

Env required:
- GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Note: Uses views if present (ads_campaign_daily, ga4_daily), otherwise falls back to base tables.
"""

import os
import sys
from datetime import date, timedelta

try:
    from google.cloud import bigquery
except Exception as e:
    print(f"ERROR: google-cloud-bigquery not available: {e}", file=sys.stderr)
    sys.exit(2)


def fetch_ads_metrics(bq: bigquery.Client, project: str, dataset: str, start: str, end: str) -> dict:
    ds = f"{project}.{dataset}"
    # Prefer daily view if exists
    has_view = False
    try:
        bq.get_table(f"{ds}.ads_campaign_daily")
        has_view = True
    except Exception:
        has_view = False
    if has_view:
        sql = f"""
            SELECT
              AVG(impression_share_p50) AS is_p50,
              SUM(impressions) AS impressions,
              SUM(clicks) AS clicks,
              SUM(conversions) AS conversions,
              SUM(cost) AS cost,
              SUM(revenue) AS revenue
            FROM `{ds}.ads_campaign_daily`
            WHERE date BETWEEN '{start}' AND '{end}'
        """
    else:
        # Aggregate from base table
        sql = f"""
            SELECT
              APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS is_p50,
              SUM(impressions) AS impressions,
              SUM(clicks) AS clicks,
              SUM(conversions) AS conversions,
              SUM(cost_micros)/1e6 AS cost,
              SUM(conversion_value) AS revenue
            FROM `{ds}.ads_campaign_performance`
            WHERE DATE(date) BETWEEN '{start}' AND '{end}'
        """
    rows = list(bq.query(sql).result())
    if not rows:
        return {}
    r = dict(rows[0])
    impressions = float(r.get('impressions') or 0)
    clicks = float(r.get('clicks') or 0)
    conversions = float(r.get('conversions') or 0)
    cost = float(r.get('cost') or 0)
    revenue = float(r.get('revenue') or 0)
    ctr = (clicks / impressions) if impressions > 0 else 0.0
    cvr = (conversions / clicks) if clicks > 0 else 0.0
    cac = (cost / conversions) if conversions > 0 else None
    roas = (revenue / cost) if cost > 0 else None
    return {
        'is_p50': float(r.get('is_p50') or 0),
        'impressions': int(impressions),
        'clicks': int(clicks),
        'conversions': int(conversions),
        'cost': float(cost),
        'revenue': float(revenue),
        'ctr': ctr,
        'cvr': cvr,
        'cac': cac,
        'roas': roas,
    }


def fetch_ga4_metrics(bq: bigquery.Client, project: str, dataset: str, start: str, end: str) -> dict:
    ds = f"{project}.{dataset}"
    # Try ga4_daily view first
    sql = None
    try:
        bq.get_table(f"{ds}.ga4_daily")
        sql = f"""
            SELECT SUM(sessions) AS sessions, SUM(conversions) AS conversions
            FROM `{ds}.ga4_daily`
            WHERE DATE(date) BETWEEN '{start}' AND '{end}'
        """
    except Exception:
        # Try raw aggregates table
        try:
            bq.get_table(f"{ds}.ga4_aggregates")
            sql = f"""
                SELECT SUM(sessions) AS sessions, SUM(conversions) AS conversions
                FROM `{ds}.ga4_aggregates`
                WHERE PARSE_DATE('%Y%m%d', date) BETWEEN '{start}' AND '{end}'
            """
        except Exception:
            return {}
    rows = list(bq.query(sql).result())
    if not rows:
        return {}
    r = dict(rows[0])
    return {
        'sessions': int(r.get('sessions') or 0),
        'conversions': int(r.get('conversions') or 0),
    }


def main():
    project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    dataset = os.environ.get('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print("Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET", file=sys.stderr)
        sys.exit(2)
    bq = bigquery.Client(project=project)

    end = date.today()
    start = end - timedelta(days=28)
    start_s, end_s = start.isoformat(), end.isoformat()

    ads = fetch_ads_metrics(bq, project, dataset, start_s, end_s)
    ga4 = fetch_ga4_metrics(bq, project, dataset, start_s, end_s)

    print("Baseline (last 28 days):")
    if ads:
        print(f"- Ads: IS p50={ads['is_p50']:.2f}, Impr={ads['impressions']:,}, Clicks={ads['clicks']:,}, Conv={ads['conversions']:,}")
        print(f"       CTR={ads['ctr']:.4f}, CVR={ads['cvr']:.4f}, CAC={'{:.2f}'.format(ads['cac']) if ads['cac'] is not None else 'n/a'}, ROAS={'{:.2f}'.format(ads['roas']) if ads['roas'] is not None else 'n/a'}")
    else:
        print("- Ads: No data in window or tables not found")
    if ga4:
        print(f"- GA4: Sessions={ga4['sessions']:,}, Conversions={ga4['conversions']:,}")
    else:
        print("- GA4: No aggregates found (ga4_daily/ga4_aggregates)")


if __name__ == '__main__':
    main()

