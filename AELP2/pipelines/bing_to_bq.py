#!/usr/bin/env python3
"""Stub: Bing Ads to BigQuery adapter (pilot)
Fetch daily performance via API (to be implemented) and write to ext tables.
"""
import os
from google.cloud import bigquery  # type: ignore

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
DATASET = os.environ.get('BIGQUERY_TRAINING_DATASET')

def main():
    if not PROJECT or not DATASET:
        print('Missing env'); return 2
    bq = bigquery.Client(project=PROJECT)
    bq.query(f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.bing_campaign_daily` (date DATE, campaign_id STRING, impressions INT64, clicks INT64, cost FLOAT64, conversions INT64, revenue FLOAT64)").result()
    print('bing_to_bq stub ready.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
