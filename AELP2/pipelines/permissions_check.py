#!/usr/bin/env python3
"""
Permissions & Accounts Checker (writes to BQ).

Checks presence of key env vars and records a status row in `<project>.<dataset>.permissions_checks`.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.permissions_checks"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('ok', 'BOOL'),
            bigquery.SchemaField('missing', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print('Missing GOOGLE_CLOUD_PROJECT/BIGQUERY_TRAINING_DATASET')
        return
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    required = ['GOOGLE_CLOUD_PROJECT', 'BIGQUERY_TRAINING_DATASET']
    optional = ['GA4_PROPERTY_ID', 'GA4_OAUTH_REFRESH_TOKEN', 'AELP2_GOOGLE_CANARY_CAMPAIGN_IDS']
    missing = [k for k in required if not os.getenv(k)]
    ok = len(missing) == 0
    row = {'timestamp': datetime.utcnow().isoformat(), 'ok': ok, 'missing': ','.join(missing)}
    bq.insert_rows_json(table_id, [row])
    print(f"Permissions check {'OK' if ok else 'MISSING:'+row['missing']}")


if __name__ == '__main__':
    main()

