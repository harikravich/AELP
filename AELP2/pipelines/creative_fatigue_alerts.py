#!/usr/bin/env python3
"""
Creative Fatigue Detection (stub): flags CTR/CVR decay over recent days.

Writes `<project>.<dataset>.creative_fatigue_alerts`.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.creative_fatigue_alerts"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('ad_id', 'STRING'),
            bigquery.SchemaField('metric', 'STRING'),
            bigquery.SchemaField('decay_pct', 'FLOAT'),
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
    # Stub: no real decay calculation; write sentinel note
    now = datetime.datetime.utcnow().isoformat()
    row = {'timestamp': now, 'ad_id': 'stub', 'metric': 'ctr', 'decay_pct': 0.0, 'notes': 'fatigue check stub'}
    bq.insert_rows_json(table_id, [row])
    print(f"Wrote fatigue alert stub to {table_id}")


if __name__ == '__main__':
    main()

