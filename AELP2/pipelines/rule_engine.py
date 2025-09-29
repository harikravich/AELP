#!/usr/bin/env python3
"""
Rule Engine (stub): evaluates safe rules and records actions (HITL required).

Writes `<project>.<dataset>.rule_engine_actions`.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.rule_engine_actions"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('rule', 'STRING'),
            bigquery.SchemaField('target', 'STRING'),
            bigquery.SchemaField('proposed_action', 'STRING'),
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
    row = {'timestamp': datetime.utcnow().isoformat(), 'rule': 'spend_spike', 'target': 'google_ads', 'proposed_action': 'pause_canary', 'shadow': True}
    bq.insert_rows_json(table_id, [row])
    print(f"Wrote rule action to {table_id}")


if __name__ == '__main__':
    main()

