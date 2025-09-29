#!/usr/bin/env python3
"""
Security Audit (enhanced stub): records IAM/audit status notes and ADC context.

Writes `<project>.<dataset>.iam_audit` with notes including ADC project/account
and presence of recommended env vars. Does not modify IAM; informational only.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.iam_audit"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('note', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    if args.dry_run and (not project or not dataset):
        print('[dry_run] Would write IAM audit note to iam_audit table; ensure least-privilege per checklist.')
        return
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    # Build note with ADC info and env presence
    note = {
        'adc_project': None,
        'adc_type': None,
        'env_present': {
            'GOOGLE_CLOUD_PROJECT': bool(os.getenv('GOOGLE_CLOUD_PROJECT')),
            'BIGQUERY_TRAINING_DATASET': bool(os.getenv('BIGQUERY_TRAINING_DATASET')),
            'GA4_PROPERTY_ID': bool(os.getenv('GA4_PROPERTY_ID')),
        },
        'recommendations': [
            'Grant least-privilege roles: bigquery.dataViewer, bigquery.jobUser on datasets',
            'Consider VPC-SC perimeter for BigQuery (optional)',
            'Rotate OAuth secrets and store outside repo',
        ],
    }
    try:
        import google.auth
        creds, proj = google.auth.default()
        note['adc_project'] = proj
        note['adc_type'] = creds.__class__.__name__
    except Exception:
        pass
    bq.insert_rows_json(table_id, [{'timestamp': datetime.utcnow().isoformat(), 'note': str(note)}])
    print(f'Wrote audit note to {table_id}')


if __name__ == '__main__':
    main()
