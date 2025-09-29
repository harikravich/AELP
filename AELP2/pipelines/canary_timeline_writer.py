#!/usr/bin/env python3
"""
Writes a simple canary timeline to BQ for dashboard consumption.
Table: `<project>.<dataset>.canary_timeline` with T-0/T-1/T-2 rows.

Idempotent, partitioned, and safe:
- DAY partitioned on `start_date`.
- Never drops the table; replaces rows for stages T-0/T-1/T-2 via DML.
- Supports `--dry_run` to avoid network access.
"""
import os
from datetime import date, timedelta
import argparse
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.canary_timeline"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('stage', 'STRING'),
            bigquery.SchemaField('start_date', 'DATE'),
            bigquery.SchemaField('notes', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        # Partition by date column
        t.time_partitioning = bigquery.TimePartitioning(field='start_date')
        bq.create_table(t)
        return table_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--t0', help='T0 date YYYY-MM-DD; default today', default=None)
    args = parser.parse_args()
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    if args.dry_run:
        print('[dry_run] would ensure table and upsert T-0/T-1/T-2 timeline rows')
        return
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    # Resolve T0 date
    if args.t0:
        y, m, d = [int(x) for x in args.t0.split('-')]
        t0 = date(y, m, d)
    else:
        t0 = date.today()
    rows = [
        {'stage': 'T-0', 'start_date': t0.isoformat(), 'notes': 'Shadow compare; no mutations'},
        {'stage': 'T-1', 'start_date': (t0 + timedelta(days=1)).isoformat(), 'notes': 'Budgets ±5%; canary ≤5%'},
        {'stage': 'T-2', 'start_date': (t0 + timedelta(days=2)).isoformat(), 'notes': 'Budgets ±10%; canary ≤10%'},
    ]
    # Idempotent replace of the three stages
    stages = ','.join([f"'{r['stage']}'" for r in rows])
    bq.query(f"DELETE FROM `{table_id}` WHERE stage IN ({stages})").result()
    bq.insert_rows_json(table_id, rows)
    print(f'Wrote canary timeline to {table_id}')


if __name__ == '__main__':
    main()
