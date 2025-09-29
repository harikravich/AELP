#!/usr/bin/env python3
"""
Post-hoc reconciliation of RL vs Ads/GA4 metrics by date.

Writes to `<project>.<dataset>.training_episodes_posthoc`:
- date, rl_spend, rl_revenue, rl_conversions, rl_roas, rl_cac
- ads_cost, ads_conversions, ads_revenue, ads_ctr, ads_cvr, ads_roas, ads_cac, ads_is_p50
- ga4_conversions (if available)
- computed_at timestamp

Usage:
  python -m AELP2.pipelines.reconcile_posthoc --start YYYY-MM-DD --end YYYY-MM-DD
"""

import os
import argparse
from datetime import datetime

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, table_id: str):
    schema = [
        bigquery.SchemaField("computed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        # RL
        bigquery.SchemaField("rl_spend", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rl_revenue", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rl_conversions", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rl_roas", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rl_cac", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rl_avg_win_rate", "FLOAT", mode="NULLABLE"),
        # Ads
        bigquery.SchemaField("ads_cost", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_conversions", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_revenue", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_ctr", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_cvr", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_roas", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_cac", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ads_is_p50", "FLOAT", mode="NULLABLE"),
        # GA4
        bigquery.SchemaField("ga4_conversions", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ga4_conversions_lagged", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
    ]
    try:
        bq.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="computed_at"
        )
        bq.create_table(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    dataset = os.environ["BIGQUERY_TRAINING_DATASET"]
    bq = bigquery.Client(project=project)
    ds = f"{project}.{dataset}"
    table_id = f"{ds}.training_episodes_posthoc"
    ensure_table(bq, table_id)
    # Reconcile schema: add ga4_conversions_lagged if missing
    tbl = bq.get_table(table_id)
    if not any(f.name == 'ga4_conversions_lagged' for f in tbl.schema):
        new_schema = list(tbl.schema) + [bigquery.SchemaField("ga4_conversions_lagged", "FLOAT", mode="NULLABLE")]
        tbl.schema = new_schema
        bq.update_table(tbl, ["schema"])  # type: ignore

    # RL daily
    rl_sql = f"""
        SELECT DATE(timestamp) AS date,
               SUM(spend) AS rl_spend,
               SUM(revenue) AS rl_revenue,
               SUM(conversions) AS rl_conversions,
               SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend),0)) AS rl_roas,
               SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions),0)) AS rl_cac,
               AVG(win_rate) AS rl_avg_win_rate
        FROM `{ds}.training_episodes`
        WHERE DATE(timestamp) BETWEEN '{args.start}' AND '{args.end}'
        GROUP BY date
    """
    rl = {str(r.date): dict(r) for r in bq.query(rl_sql).result()}

    # Ads daily
    ads_sql = f"""
        SELECT DATE(date) AS date,
               SUM(cost_micros)/1e6 AS ads_cost,
               SUM(conversions) AS ads_conversions,
               SUM(conversion_value) AS ads_revenue,
               SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ads_ctr,
               SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS ads_cvr,
               SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS ads_roas,
               SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS ads_cac,
               APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS ads_is_p50
        FROM `{ds}.ads_campaign_performance`
        WHERE DATE(date) BETWEEN '{args.start}' AND '{args.end}'
        GROUP BY date
    """
    ads = {str(r.date): dict(r) for r in bq.query(ads_sql).result()}

    # GA4 daily (optional)
    ga4 = {}
    ga4_lag = {}
    try:
        ga4_sql = f"""
            SELECT DATE(date) AS date, SUM(conversions) AS ga4_conversions
            FROM `{ds}.ga4_daily`
            WHERE DATE(date) BETWEEN '{args.start}' AND '{args.end}'
            GROUP BY date
        """
        ga4 = {str(r.date): dict(r) for r in bq.query(ga4_sql).result()}
    except Exception:
        ga4 = {}
    try:
        ga4_lag_sql = f"""
            SELECT DATE(date) AS date, SUM(ga4_conversions_lagged) AS ga4_conversions_lagged
            FROM `{ds}.ga4_lagged_attribution`
            WHERE DATE(date) BETWEEN '{args.start}' AND '{args.end}'
            GROUP BY date
        """
        ga4_lag = {str(r.date): dict(r) for r in bq.query(ga4_lag_sql).result()}
    except Exception:
        ga4_lag = {}

    # Build rows for common dates
    dates = sorted(set(rl.keys()) | set(ads.keys()))
    out = []
    now = datetime.utcnow().isoformat()
    for d in dates:
        r = rl.get(d, {})
        a = ads.get(d, {})
        g = ga4.get(d, {})
        gl = ga4_lag.get(d, {})
        row = {
            'computed_at': now,
            'date': d,
            'rl_spend': float(r.get('rl_spend') or 0.0),
            'rl_revenue': float(r.get('rl_revenue') or 0.0),
            'rl_conversions': float(r.get('rl_conversions') or 0.0),
            'rl_roas': float(r.get('rl_roas') or 0.0) if r.get('rl_roas') is not None else None,
            'rl_cac': float(r.get('rl_cac') or 0.0) if r.get('rl_cac') is not None else None,
            'rl_avg_win_rate': float(r.get('rl_avg_win_rate') or 0.0),
            'ads_cost': float(a.get('ads_cost') or 0.0),
            'ads_conversions': float(a.get('ads_conversions') or 0.0),
            'ads_revenue': float(a.get('ads_revenue') or 0.0),
            'ads_ctr': float(a.get('ads_ctr') or 0.0) if a.get('ads_ctr') is not None else None,
            'ads_cvr': float(a.get('ads_cvr') or 0.0) if a.get('ads_cvr') is not None else None,
            'ads_roas': float(a.get('ads_roas') or 0.0) if a.get('ads_roas') is not None else None,
            'ads_cac': float(a.get('ads_cac') or 0.0) if a.get('ads_cac') is not None else None,
            'ads_is_p50': float(a.get('ads_is_p50') or 0.0) if a.get('ads_is_p50') is not None else None,
            'ga4_conversions': float(g.get('ga4_conversions') or 0.0) if g else None,
            'ga4_conversions_lagged': float(gl.get('ga4_conversions_lagged') or 0.0) if gl else None,
            'details': None,
        }
        out.append(row)

    # Insert rows
    if out:
        bq.insert_rows_json(table_id, out)
        print(f"Inserted {len(out)} rows into {table_id}")
    else:
        print("No rows to write (no overlap between RL and Ads in window)")


if __name__ == '__main__':
    main()
