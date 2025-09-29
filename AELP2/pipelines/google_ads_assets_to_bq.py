#!/usr/bin/env python3
"""
Google Ads Assets ingestion.

Ensures `<project>.<dataset>.ads_assets` table and fetches basic asset metadata
(TEXT/YOUTUBE_VIDEO/IMAGE). Stores text (for TEXT assets), youtube_id, image_url,
and policy summary if available.
"""
import os
from datetime import datetime
from typing import List, Dict

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
try:
    from google.ads.googleads.client import GoogleAdsClient  # type: ignore
    from google.ads.googleads.errors import GoogleAdsException  # type: ignore
    ADS_AVAILABLE = True
except Exception:
    ADS_AVAILABLE = False


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.ads_assets"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('asset_id', 'STRING'),
            bigquery.SchemaField('type', 'STRING'),
            bigquery.SchemaField('text', 'STRING'),
            bigquery.SchemaField('youtube_id', 'STRING'),
            bigquery.SchemaField('image_url', 'STRING'),
            bigquery.SchemaField('policy_summary', 'STRING'),
            bigquery.SchemaField('fetched_at', 'TIMESTAMP'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='fetched_at')
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
        print('[dry_run] Would ensure ads_assets table and fetch assets when creds available.')
        return
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    if not ADS_AVAILABLE:
        print('google-ads not installed; assets table ensured; skipping fetch')
        return
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
    ga_service = client.get_service('GoogleAdsService')
    customer_id = os.environ['GOOGLE_ADS_CUSTOMER_ID'].replace('-', '')
    query = (
        "SELECT asset.resource_name, asset.type, asset.name, "
        "asset.text_asset.text, asset.youtube_video_asset.youtube_video_id, "
        "asset.image_asset.full_size_url, asset.policy_summary.approval_status "
        "FROM asset LIMIT 5000"
    )
    try:
        response = ga_service.search(customer_id=customer_id, query=query)
    except GoogleAdsException as e:
        print(f"Google Ads error: {e}")
        return
    rows=[]
    for r in response:
        a=r.asset
        rows.append({
            'asset_id': a.resource_name,
            'type': str(a.type_.name) if hasattr(a.type_, 'name') else str(a.type_),
            'text': getattr(a.text_asset, 'text', None) if a.text_asset else None,
            'youtube_id': getattr(a.youtube_video_asset, 'youtube_video_id', None) if a.youtube_video_asset else None,
            'image_url': getattr(a.image_asset, 'full_size_url', None) if a.image_asset else None,
            'policy_summary': str(getattr(a.policy_summary, 'approval_status', '')),
            'fetched_at': datetime.utcnow().isoformat()
        })
    if rows:
        job = bq.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_APPEND'))
        job.result()
        print(f'Loaded {len(rows)} assets into {table_id}')
    else:
        print('No assets returned')


if __name__ == '__main__':
    main()
