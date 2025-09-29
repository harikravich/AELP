#!/usr/bin/env python3
"""
RL Policy Hints Writer (bootstrap)

Ensures the `policy_hints` table exists and provides a simple CLI to write
hint rows produced by the RL lab (offline). This does not run RL; it’s a bridge.

Schema fields:
- timestamp TIMESTAMP
- source STRING (e.g., 'rl_lab')
- hint_type STRING ('exploration_set' | 'budget_tilt' | 'opportunity')
- payload JSON (freeform details)
- notes STRING
"""
import os
import json
from datetime import datetime
from typing import Dict, Any

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.policy_hints"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        pass
    schema = [
        bigquery.SchemaField('timestamp', 'TIMESTAMP', mode='REQUIRED'),
        bigquery.SchemaField('source', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('hint_type', 'STRING', mode='REQUIRED'),
        bigquery.SchemaField('payload', 'JSON', mode='REQUIRED'),
        bigquery.SchemaField('notes', 'STRING', mode='NULLABLE'),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
    bq.create_table(table)
    return table_id


def write_hint(bq: bigquery.Client, table_id: str, source: str, hint_type: str, payload: Dict[str, Any], notes: str = None):
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'source': source,
        'hint_type': hint_type,
        'payload': json.dumps(payload),
        'notes': notes,
    }
    bq.insert_rows_json(table_id, [row])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='rl_lab')
    p.add_argument('--hint_type', required=True, choices=['exploration_set', 'budget_tilt', 'opportunity'])
    p.add_argument('--payload_json', required=True, help='JSON string payload')
    p.add_argument('--notes', default=None)
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    payload = json.loads(args.payload_json)
    write_hint(bq, table_id, args.source, args.hint_type, payload, args.notes)
    print(f"Wrote policy_hint of type {args.hint_type} to {table_id}")


if __name__ == '__main__':
    main()

