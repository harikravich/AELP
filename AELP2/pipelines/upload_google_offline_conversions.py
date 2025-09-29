#!/usr/bin/env python3
"""
Google Offline Conversions Upload (HITL-gated):

Builds payloads with hashed PII for Enhanced Conversions and logs an entry to
`<project>.<dataset>.value_uploads_log`. If `AELP2_ALLOW_VALUE_UPLOADS=1` and
`AELP2_VALUE_UPLOAD_DRY_RUN=0`, will attempt real API call via google-ads.
Otherwise logs intent only.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from AELP2.adapters.google_enhanced_conversions import build_enhanced_conversions


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
    # Build example payload from staging table if present
    payload_rows = []
    try:
        [rows] = bq.query({ 'query': f"SELECT * FROM `{project}.{dataset}.value_uploads_staging` WHERE DATE(timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) LIMIT 100" })  # type: ignore
    except Exception:
        rows = []
    try:
        payload_rows = build_enhanced_conversions([dict(r) for r in rows])
    except Exception:
        payload_rows = []
    # Optional real API call (guarded)
    dry = os.getenv('AELP2_VALUE_UPLOAD_DRY_RUN', '1') == '1'
    api_note = 'dry_run'
    if allow and not dry and payload_rows:
        try:
            from google.ads.googleads.client import GoogleAdsClient  # type: ignore
            cfg = {
                'developer_token': os.environ['GOOGLE_ADS_DEVELOPER_TOKEN'],
                'client_id': os.environ['GOOGLE_ADS_CLIENT_ID'],
                'client_secret': os.environ['GOOGLE_ADS_CLIENT_SECRET'],
                'refresh_token': os.environ['GOOGLE_ADS_REFRESH_TOKEN'],
                'use_proto_plus': True,
            }
            login_cid = os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
            if login_cid:
                cfg['login_customer_id'] = login_cid.replace('-', '')
            client = GoogleAdsClient.load_from_dict(cfg)
            conv_upload = client.get_service('ConversionUploadService')
            # Build UploadClickConversionsRequest with user identifiers (enhanced conversions)
            conversions = []
            for p in payload_rows:
                c = client.get_type('ClickConversion')
                # Enhanced conversions via user_identifiers
                for k, v in (p.get('user_identifiers') or {}).items():
                    if v is None:
                        continue
                    ui = client.get_type('UserIdentifier')
                    if k == 'hashed_email':
                        ui.hashed_email = v
                    elif k == 'hashed_phone_number':
                        ui.hashed_phone_number = v
                    elif k == 'address_info':
                        info = client.get_type('OfflineUserAddressInfo')
                        a = v or {}
                        if a.get('hashed_first_name'): info.hashed_first_name = a['hashed_first_name']
                        if a.get('hashed_last_name'): info.hashed_last_name = a['hashed_last_name']
                        if a.get('country_code'): info.country_code = a['country_code']
                        if a.get('postal_code'): info.postal_code = a['postal_code']
                        ui.address_info = info
                    c.user_identifiers.append(ui)
                c.conversion_action = p.get('conversion_action')
                c.conversion_value = float(p.get('conversion_value') or p.get('value') or 0.0)
                c.currency_code = p.get('currency_code') or p.get('currency') or 'USD'
                if p.get('order_id'):
                    c.order_id = p['order_id']
                conversions.append(c)
            req = client.get_type('UploadClickConversionsRequest')
            req.customer_id = (os.environ.get('GOOGLE_ADS_CUSTOMER_ID') or '').replace('-', '')
            req.conversions.extend(conversions)
            req.partial_failure = True
            resp = conv_upload.upload_click_conversions(request=req)
            api_note = f"uploaded {len(resp.results)} conversions; partial_failure={bool(resp.partial_failure_error)}"
        except Exception as e:
            api_note = f"upload_error: {e}"
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'platform': 'google_ads',
        'payload_ref': ref,
        'rows': len(payload_rows) or int(os.getenv('AELP2_VALUE_UPLOAD_ROWS', '0') or '0'),
        'allow_uploads': allow and not dry,
        'notes': api_note,
    }
    bq.insert_rows_json(table_id, [row])
    print(f"Logged Google value upload intent to {table_id}: {row}")


if __name__ == '__main__':
    main()
