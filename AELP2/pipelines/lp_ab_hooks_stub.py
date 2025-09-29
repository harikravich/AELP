#!/usr/bin/env python3
"""
Landing-Page A/B Hooks (stub): propose UTM cohorts and GA4 goal names.

Writes `<project>.<dataset>.lp_ab_candidates` with shadow-only proposals.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.lp_ab_candidates"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('url', 'STRING'),
            bigquery.SchemaField('utm', 'STRING'),
            bigquery.SchemaField('ga4_goal', 'STRING'),
            bigquery.SchemaField('notes', 'STRING'),
        ]
        bq.create_table(bigquery.Table(table_id, schema=schema))
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    now = datetime.utcnow().isoformat()
    rows = [{
        'timestamp': now,
        'url': 'https://example.com/landing?utm_source=google&utm_campaign=ab_test_A',
        'utm': 'utm_source=google&utm_campaign=ab_test_A',
        'ga4_goal': 'sign_up',
        'notes': 'stub candidate; implement GA4 goal hooks and cohorting',
    }]
    bq.insert_rows_json(table_id, rows)
    print(f'Wrote {len(rows)} landing-page A/B candidates to {table_id}')


if __name__ == '__main__':
    main()

