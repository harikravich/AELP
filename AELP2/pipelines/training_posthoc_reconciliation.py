#!/usr/bin/env python3
"""
Post-hoc reconciliation (lag-aware KPIs).

Writes `<project>.<dataset>.training_episodes_posthoc` by joining Ads spend with
GA4 lagged conversions for each date, computing lag-aware CAC/ROAS. Safe if inputs
missing: ensures table and exits with a note.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List, Dict

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.training_episodes_posthoc"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('cost', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('ga4_conversions_lagged', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('revenue', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('cac', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('roas', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('details', 'JSON', mode='NULLABLE'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    # Check if views exist
    try:
        bq.get_table(f"{project}.{dataset}.ads_campaign_daily")
    except NotFound:
        print('ads_campaign_daily view not found; run create_bq_views.py first')
        return
    try:
        bq.get_table(f"{project}.{dataset}.ga4_lagged_daily")
    except NotFound:
        print('ga4_lagged_daily view not found; compute ga4_lagged_attribution first')
        return

    sql = f"""
      WITH ads AS (
        SELECT DATE(date) AS date, SUM(cost) AS cost
        FROM `{project}.{dataset}.ads_campaign_daily`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        GROUP BY date
      ),
      ga4 AS (
        SELECT DATE(date) AS date, SUM(ga4_conversions_lagged) AS ga4_conversions_lagged
        FROM `{project}.{dataset}.ga4_lagged_daily`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        GROUP BY date
      )
      SELECT a.date,
             a.cost,
             g.ga4_conversions_lagged,
             -- revenue proxy via env AELP2_AOV or AELP2_LTV if present
             (g.ga4_conversions_lagged * CAST(COALESCE(NULLIF(@AELP2_LTV, ''), NULLIF(@AELP2_AOV, '')) AS FLOAT64)) AS revenue,
             SAFE_DIVIDE(a.cost, NULLIF(g.ga4_conversions_lagged,0)) AS cac,
             SAFE_DIVIDE((g.ga4_conversions_lagged * CAST(COALESCE(NULLIF(@AELP2_LTV, ''), NULLIF(@AELP2_AOV, '')) AS FLOAT64)), NULLIF(a.cost,0)) AS roas
      FROM ads a
      LEFT JOIN ga4 g USING(date)
      ORDER BY a.date
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('AELP2_AOV', 'STRING', os.getenv('AELP2_AOV', '100')),
        bigquery.ScalarQueryParameter('AELP2_LTV', 'STRING', os.getenv('AELP2_LTV', '')),
    ]))
    rows = [dict(r) for r in job.result()]
    if not rows:
        print('No post-hoc rows computed')
        return
    out: List[Dict] = []
    for r in rows:
        out.append({
            'date': r['date'].isoformat(),
            'cost': float(r.get('cost') or 0.0),
            'ga4_conversions_lagged': float(r.get('ga4_conversions_lagged') or 0.0),
            'revenue': float(r.get('revenue') or 0.0),
            'cac': float(r.get('cac') or 0.0),
            'roas': float(r.get('roas') or 0.0),
            'details': None,
        })
    bq.insert_rows_json(table_id, out)
    print(f"Inserted {len(out)} post-hoc rows into {table_id}")


if __name__ == '__main__':
    main()

