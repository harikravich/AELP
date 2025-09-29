#!/usr/bin/env python3
"""
Channel Attribution Runner (skeleton)

Intended to run an R-based ChannelAttribution (Markov/Shapley) job weekly and write summaries
to BigQuery. If R/deps are unavailable, writes a placeholder summary and exits 0.
"""
import os
from datetime import date, timedelta
from google.cloud import bigquery


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.channel_attribution_weekly"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('week_start', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('channel', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('method', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('conversions', 'FLOAT', mode='REQUIRED'),
            bigquery.SchemaField('revenue', 'FLOAT', mode='REQUIRED'),
            bigquery.SchemaField('notes', 'STRING', mode='NULLABLE'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='week_start')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    # Placeholder row
    start = date.today() - timedelta(days=7)
    rows = [
        {'week_start': start.isoformat(), 'channel': 'google_ads', 'method': 'placeholder', 'conversions': 0.0, 'revenue': 0.0, 'notes': 'Runner scaffold; install R job'},
    ]
    bq.insert_rows_json(table_id, rows)
    print(f"Wrote placeholder channel attribution summary to {table_id}")


if __name__ == '__main__':
    main()

