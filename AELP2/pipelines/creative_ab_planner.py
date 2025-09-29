#!/usr/bin/env python3
"""
Creative AB Planner (shadow-only): proposes a test and writes to BQ.

Tables:
- `<project>.<dataset>.ab_experiments`
- `<project>.<dataset>.creative_variants`
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_tables(bq: bigquery.Client, project: str, dataset: str):
    exp = f"{project}.{dataset}.ab_experiments"
    var = f"{project}.{dataset}.creative_variants"
    try:
        bq.get_table(exp)
    except NotFound:
        schema = [
            bigquery.SchemaField('experiment_id', 'STRING'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('start', 'TIMESTAMP'),
            bigquery.SchemaField('end', 'TIMESTAMP'),
            bigquery.SchemaField('status', 'STRING'),
        ]
        t = bigquery.Table(exp, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='start')
        bq.create_table(t)
    try:
        bq.get_table(var)
    except NotFound:
        schema = [
            bigquery.SchemaField('variant_id', 'STRING'),
            bigquery.SchemaField('experiment_id', 'STRING'),
            bigquery.SchemaField('gen_method', 'STRING'),
            bigquery.SchemaField('text', 'STRING'),
            bigquery.SchemaField('policy_flags', 'STRING'),
            bigquery.SchemaField('created', 'TIMESTAMP'),
        ]
        t = bigquery.Table(var, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='created')
        bq.create_table(t)
    return exp, var


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    exp_tbl, var_tbl = ensure_tables(bq, project, dataset)
    ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    exp_id = f'exp_{ts}'
    v1 = f'v1_{ts}'
    v2 = f'v2_{ts}'
    bq.insert_rows_json(exp_tbl, [{
        'experiment_id': exp_id,
        'platform': 'google_ads',
        'campaign_id': 'canary',
        'start': datetime.utcnow().isoformat(),
        'end': None,
        'status': 'proposed',
    }])
    bq.insert_rows_json(var_tbl, [
        {'variant_id': v1, 'experiment_id': exp_id, 'gen_method': 'baseline', 'text': 'Control text', 'policy_flags': '', 'created': datetime.utcnow().isoformat()},
        {'variant_id': v2, 'experiment_id': exp_id, 'gen_method': 'copy_opt_stub', 'text': 'Variant text', 'policy_flags': '', 'created': datetime.utcnow().isoformat()},
    ])
    print(f'Wrote AB experiment {exp_id} with 2 variants')


if __name__ == '__main__':
    main()
