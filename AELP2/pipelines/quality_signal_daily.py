#!/usr/bin/env python3
"""
Quality Signal Daily (stub): computes a simple quality proxy (trial→paid, retention).

Writes `<project>.<dataset>.quality_signal_daily` with safe defaults if inputs are missing.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.quality_signal_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('trial_to_paid', 'FLOAT'),
            bigquery.SchemaField('retention_7d', 'FLOAT'),
            bigquery.SchemaField('note', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        return table_id


def main():
    from datetime import date
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    row = {'date': date.today().isoformat(), 'trial_to_paid': 0.0, 'retention_7d': 0.0, 'note': 'stub default'}
    bq.insert_rows_json(table_id, [row])
    print(f'Wrote quality signal to {table_id}')


if __name__ == '__main__':
    main()

