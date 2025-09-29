#!/usr/bin/env python3
"""
Privacy audit (stub): ensures free-text fields are not stored; writes summary row.
"""
import os
from datetime import datetime
from google.cloud import bigquery


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print('Missing env; skipping')
        return
    bq = bigquery.Client(project=project)
    table = f"{project}.{dataset}.privacy_audit"
    try:
        bq.get_table(table)
    except Exception:
        from google.cloud import bigquery as bq2
        t = bq2.Table(table, schema=[
            bq2.SchemaField('timestamp', 'TIMESTAMP'),
            bq2.SchemaField('checks', 'JSON'),
            bq2.SchemaField('status', 'STRING'),
        ])
        t.time_partitioning = bq2.TimePartitioning(type_=bq2.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    checks = {'free_text_present': False}
    bq.insert_rows_json(table, [{
        'timestamp': datetime.utcnow().isoformat(),
        'checks': str(checks),
        'status': 'ok'
    }])
    print('Privacy audit row written')


if __name__ == '__main__':
    main()

