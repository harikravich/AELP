#!/usr/bin/env python3
import os
import json
from datetime import datetime
from typing import Optional
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import argparse


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.platform_skeletons"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("platform", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("campaign_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("objective", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("daily_budget", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("notes", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("utm", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="timestamp")
        bq.create_table(table)
        return table_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--platform', required=True)
    p.add_argument('--campaign_name', required=True)
    p.add_argument('--objective', default='lead_generation')
    p.add_argument('--daily_budget', type=float, default=50.0)
    p.add_argument('--notes', default='paused skeleton')
    p.add_argument('--utm', default='{}')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'platform': args.platform,
        'campaign_name': args.campaign_name,
        'objective': args.objective,
        'daily_budget': float(args.daily_budget),
        'status': 'paused',
        'notes': args.notes,
        'utm': args.utm,
    }
    bq.insert_rows_json(table_id, [row])
    print(f"Logged skeleton: {row}")


if __name__ == '__main__':
    main()

