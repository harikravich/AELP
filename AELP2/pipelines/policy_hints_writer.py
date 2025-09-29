#!/usr/bin/env python3
"""
Policy Hints writer (stub): writes exploration/budget tilt hints to BQ.

Writes `<project>.<dataset>.policy_hints` with sample hint rows (shadow).
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.policy_hints"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('hint_type', 'STRING'),
            bigquery.SchemaField('target', 'STRING'),
            bigquery.SchemaField('value', 'FLOAT'),
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
    rows = [
        {'timestamp': now, 'hint_type': 'budget_tilt', 'target': 'google_ads:search', 'value': 0.05, 'shadow': True},
        {'timestamp': now, 'hint_type': 'exploration', 'target': 'creative', 'value': 0.10, 'shadow': True},
    ]
    bq.insert_rows_json(table_id, rows)
    print(f'Inserted {len(rows)} policy hints into {table_id}')


if __name__ == '__main__':
    main()

