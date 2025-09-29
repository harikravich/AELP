#!/usr/bin/env python3
"""
Cross-platform Budget Broker (shadow-only, stub):

Aggregates proposed allocations (from MMM or portfolio) and produces a
`broker_allocations` table keyed by platform and campaign_id with shares.

Inputs (if present):
- `<project>.<dataset>.portfolio_allocations` (preferred)
- `<project>.<dataset>.mmm_allocations` (fallback single-channel)

Output:
- `<project>.<dataset>.broker_allocations`
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.broker_allocations"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('proposed_budget', 'FLOAT'),
            bigquery.SchemaField('share', 'FLOAT'),
            bigquery.SchemaField('source', 'STRING'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    # Prefer portfolio_allocations
    src = 'portfolio_allocations'
    try:
        bq.get_table(f"{project}.{dataset}.portfolio_allocations")
        sql = f"""
          SELECT 'google_ads' AS platform, CAST(campaign_id AS STRING) AS campaign_id,
                 proposed_budget, share
          FROM `{project}.{dataset}.portfolio_allocations`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY)
        """
    except Exception:
        src = 'mmm_allocations'
        sql = f"""
          SELECT 'google_ads' AS platform, channel AS campaign_id,
                 proposed_daily_budget AS proposed_budget,
                 1.0 AS share
          FROM `{project}.{dataset}.mmm_allocations`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          LIMIT 1
        """
    rows = [dict(r) for r in bq.query(sql).result()]
    now = datetime.utcnow().isoformat()
    out = []
    for r in rows:
        out.append({
            'timestamp': now,
            'platform': r['platform'],
            'campaign_id': str(r['campaign_id']),
            'proposed_budget': float(r['proposed_budget'] or 0.0),
            'share': float(r['share'] or 0.0),
            'source': src,
            'shadow': True,
        })
    if out:
        bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} broker rows to {table_id}")


if __name__ == '__main__':
    main()

