#!/usr/bin/env python3
"""
Meta CAPI Conversions Upload (HITL-gated): builds hashed payload and logs intent.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import json
import urllib.request
from AELP2.adapters.meta_capi import build_capi_events


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.value_uploads_log"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('payload_ref', 'STRING'),
            bigquery.SchemaField('rows', 'INT64'),
            bigquery.SchemaField('allow_uploads', 'BOOL'),
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
    allow = os.getenv('AELP2_ALLOW_VALUE_UPLOADS', '0') == '1'
    ref = os.getenv('AELP2_VALUE_UPLOAD_PAYLOAD_REF', 'stub')
    # Build payload from staging if present
    payload_rows = []
    try:
        [rows] = bq.query({ 'query': f"SELECT * FROM `{project}.{dataset}.value_uploads_staging` WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) LIMIT 100" })  # type: ignore
    except Exception:
        rows = []
    try:
        payload_rows = build_capi_events([dict(r) for r in rows])
    except Exception:
        payload_rows = []
    dry = os.getenv('AELP2_VALUE_UPLOAD_DRY_RUN', '1') == '1'
    api_note = 'dry_run'
    if allow and not dry and payload_rows:
        try:
            pixel_id = os.environ['META_PIXEL_ID']
            token = os.environ['META_ACCESS_TOKEN']
            url = f'https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}'
            body = json.dumps({'data': payload_rows}).encode('utf-8')
            req = urllib.request.Request(url, data=body, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req, timeout=10) as r:
                resp = r.read().decode('utf-8')
                api_note = f"uploaded:{resp[:120]}..."
        except Exception as e:
            api_note = f"upload_error: {e}"
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'platform': 'meta',
        'payload_ref': ref,
        'rows': len(payload_rows) or int(os.getenv('AELP2_VALUE_UPLOAD_ROWS', '0') or '0'),
        'allow_uploads': allow and not dry,
        'notes': api_note,
    }
    bq.insert_rows_json(table_id, [row])
    print(f"Logged Meta value upload intent to {table_id}: {row}")


if __name__ == '__main__':
    main()
