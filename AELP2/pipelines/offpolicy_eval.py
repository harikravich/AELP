#!/usr/bin/env python3
"""
Off-policy evaluation (stub): compares hints vs realized outcomes.

Writes `<project>.<dataset>.offpolicy_eval_results` with placeholder metrics.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.offpolicy_eval_results"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('metric', 'STRING'),
            bigquery.SchemaField('value', 'FLOAT'),
            bigquery.SchemaField('notes', 'STRING'),
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
        {'timestamp': now, 'metric': 'expected_roas_lift', 'value': 0.0, 'notes': 'stub'},
        {'timestamp': now, 'metric': 'expected_cac_delta', 'value': 0.0, 'notes': 'stub'},
    ]
    bq.insert_rows_json(table_id, rows)
    print(f'Inserted {len(rows)} off-policy metrics into {table_id}')


if __name__ == '__main__':
    main()

