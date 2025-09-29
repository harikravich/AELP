#!/usr/bin/env python3
"""
Real-time budget pacer (stub): emits pacing proposals at minute cadence.

Writes `<project>.<dataset>.budget_pacing_proposals`.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.budget_pacing_proposals"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('delta_pct', 'FLOAT'),
            bigquery.SchemaField('notes', 'STRING'),
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
    now = datetime.utcnow().isoformat()
    row = {'timestamp': now, 'campaign_id': 'all', 'delta_pct': 0.0, 'notes': 'pacer stub', 'shadow': True}
    bq.insert_rows_json(table_id, [row])
    print(f"Wrote pacing proposal to {table_id}")


if __name__ == '__main__':
    main()

