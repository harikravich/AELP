#!/usr/bin/env python3
"""
Dayparting Optimizer (stub): propose hour/day schedule caps.

Writes `<project>.<dataset>.dayparting_schedules` with safe default caps.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.dayparting_schedules"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('dow', 'INT64'),
            bigquery.SchemaField('hour', 'INT64'),
            bigquery.SchemaField('budget_multiplier', 'FLOAT'),
            bigquery.SchemaField('notes', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    import datetime
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    now = datetime.datetime.utcnow().isoformat()
    # Default flat schedule multipliers (1.0)
    out = []
    for dow in range(7):
        for hour in range(24):
            out.append({'timestamp': now, 'campaign_id': 'all', 'dow': dow, 'hour': hour, 'budget_multiplier': 1.0, 'notes': 'flat schedule (stub)'})
    bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} dayparting rows to {table_id}")


if __name__ == '__main__':
    main()

