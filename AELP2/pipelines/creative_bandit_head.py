#!/usr/bin/env python3
"""
Creative Bandit Head (stub): logs decision proposals to ab_experiments (shadow-only).
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
    exp = f"{project}.{dataset}.ab_experiments"
    try:
        bq.get_table(exp)
    except Exception:
        from google.cloud import bigquery as bq2
        t = bq2.Table(exp, schema=[
            bq2.SchemaField('experiment_id','STRING'),
            bq2.SchemaField('platform','STRING'),
            bq2.SchemaField('campaign_id','STRING'),
            bq2.SchemaField('start','TIMESTAMP'),
            bq2.SchemaField('end','TIMESTAMP'),
            bq2.SchemaField('status','STRING'),
        ])
        t.time_partitioning = bq2.TimePartitioning(type_=bq2.TimePartitioningType.DAY, field='start')
        bq.create_table(t)
    eid = f"bandit_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    bq.insert_rows_json(exp, [{
        'experiment_id': eid,
        'platform': 'google_ads',
        'campaign_id': 'canary',
        'start': datetime.utcnow().isoformat(),
        'end': None,
        'status': 'proposed',
    }])
    print('Bandit proposal logged (shadow)')


if __name__ == '__main__':
    main()

