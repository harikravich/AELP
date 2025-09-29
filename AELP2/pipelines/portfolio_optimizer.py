#!/usr/bin/env python3
"""
Portfolio Optimizer (stub): propose daily cross-campaign allocations under CAC cap.

Writes `<project>.<dataset>.portfolio_allocations` with proposed budget shares.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.portfolio_allocations"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('proposed_budget', 'FLOAT'),
            bigquery.SchemaField('share', 'FLOAT'),
            bigquery.SchemaField('notes', 'STRING'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    import json, datetime
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS campaign_id,
             SUM(cost_micros)/1e6 AS cost,
             SUM(conversions) AS conv
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      GROUP BY campaign_id
      HAVING SUM(cost_micros) > 0
      ORDER BY cost DESC
      LIMIT 20
    """
    try:
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows = []
    if not rows:
        print('No campaigns found; wrote 0 allocations')
        return
    total_cost = sum(float(r['cost'] or 0.0) for r in rows) or 1.0
    now = datetime.datetime.utcnow().isoformat()
    out = []
    for r in rows:
        share = float(r['cost']) / total_cost
        out.append({
            'timestamp': now,
            'campaign_id': r['campaign_id'],
            'proposed_budget': max(10.0, 1000.0 * share),
            'share': share,
            'notes': 'proportional-to-spend (stub)',
            'shadow': True,
        })
    bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} portfolio allocations to {table_id}")


if __name__ == '__main__':
    main()

