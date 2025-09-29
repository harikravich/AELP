#!/usr/bin/env python3
"""
Value-Based Bidding Bridge (stubs)

Prepares offline conversion payloads with predicted values for upload to ads platforms.
This is a stub that assembles payloads from BQ and prints or writes to staging table.

Note: Actual upload requires platform-specific SDKs and consented hashing; we gate that
behind HITL and separate scripts.
"""
import os
import json
from datetime import date, timedelta
from google.cloud import bigquery


def fetch_recent_conversions(bq: bigquery.Client, project: str, dataset: str, days: int = 7):
    # Placeholder: expect a table of recent conversions with identifiers and predicted value
    sql = f"""
      SELECT
        DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY) AS conversion_date,
        'gclid_sample' AS gclid,
        123.45 AS value,
        'USD' AS currency
    """
    return [dict(r) for r in bq.query(sql).result()]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    rows = fetch_recent_conversions(bq, project, dataset)
    payload = [{'gclid': r['gclid'], 'conversion_date_time': str(r['conversion_date']), 'value': r['value'], 'currency_code': r['currency']} for r in rows]
    if args.dry_run:
        print(json.dumps({'offline_conversions': payload}, indent=2))
        return
    # In production, write to a staging table for uploader jobs
    table_id = f"{project}.{dataset}.offline_conversions_staging"
    schema = [
        bigquery.SchemaField('gclid', 'STRING'),
        bigquery.SchemaField('conversion_date_time', 'STRING'),
        bigquery.SchemaField('value', 'FLOAT'),
        bigquery.SchemaField('currency_code', 'STRING'),
    ]
    try:
        bq.get_table(table_id)
    except Exception:
        bq.create_table(bigquery.Table(table_id, schema=schema))
    bq.insert_rows_json(table_id, payload)
    print(f"Wrote {len(payload)} offline conversions to {table_id} (staging)")


if __name__ == '__main__':
    main()

