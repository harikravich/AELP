#!/usr/bin/env python3
"""
Competitive Intelligence ingest (stub): ensure auction insights table exists.

Writes `<project>.<dataset>.ads_auction_insights` schema only if missing.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.ads_auction_insights"
    try:
        bq.get_table(table_id)
        print('ads_auction_insights exists')
        return
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('customer_id', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('overlap_rate', 'FLOAT'),
            bigquery.SchemaField('position_above_rate', 'FLOAT'),
            bigquery.SchemaField('top_of_page_rate', 'FLOAT'),
            bigquery.SchemaField('abs_top_of_page_rate', 'FLOAT'),
            bigquery.SchemaField('outranking_share', 'FLOAT'),
            bigquery.SchemaField('domain', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        print('ads_auction_insights created (empty)')


if __name__ == '__main__':
    main()

