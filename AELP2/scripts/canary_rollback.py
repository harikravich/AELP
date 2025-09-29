#!/usr/bin/env python3
"""
Canary rollback (shadow-only): writes rollback intents into BigQuery.

Reads last N rows from `<project>.<dataset>.canary_changes` and creates
`<project>.<dataset>.canary_rollbacks` entries reverting budgets to old_budget.
No live mutations are performed.
"""
import os
import argparse
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.canary_rollbacks"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('rollback_to_budget', 'FLOAT'),
            bigquery.SchemaField('notes', 'STRING'),
            bigquery.SchemaField('applied', 'BOOL'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--last_n', type=int, default=5)
    args = p.parse_args()
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    try:
        sql = f"""
          SELECT campaign_id, old_budget
          FROM `{project}.{dataset}.canary_changes`
          ORDER BY timestamp DESC
          LIMIT {args.last_n}
        """
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows = []
    now = datetime.utcnow().isoformat()
    out = []
    for r in rows:
        out.append({
            'timestamp': now,
            'campaign_id': str(r.get('campaign_id', '')),
            'rollback_to_budget': float(r.get('old_budget') or 0.0),
            'notes': 'shadow rollback intent; no live mutations',
            'applied': False,
            'shadow': True,
        })
    if out:
        bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} rollback intents to {table_id}")


if __name__ == '__main__':
    main()

