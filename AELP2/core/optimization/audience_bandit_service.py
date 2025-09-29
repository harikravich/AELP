#!/usr/bin/env python3
"""
Audience Bandit Service (shadow): Thompson Sampling over audience segments.

Arms: segments from `<project>.<dataset>.segment_scores_daily` with proxy successes
based on positive uplift score. Dry-run uses synthetic segments.
Writes decisions to `bandit_decisions` (platform='audience', channel='segments').
"""
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.bandit_decisions"
    try:
        bq.get_table(table_id)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp','TIMESTAMP'),
            bigquery.SchemaField('platform','STRING'),
            bigquery.SchemaField('channel','STRING'),
            bigquery.SchemaField('campaign_id','STRING'),
            bigquery.SchemaField('ad_id','STRING'),
            bigquery.SchemaField('prior_alpha','FLOAT'),
            bigquery.SchemaField('prior_beta','FLOAT'),
            bigquery.SchemaField('posterior_alpha','FLOAT'),
            bigquery.SchemaField('posterior_beta','FLOAT'),
            bigquery.SchemaField('sample','FLOAT'),
            bigquery.SchemaField('context','JSON'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    return table_id


def fetch_segments(bq: bigquery.Client, project: str, dataset: str) -> List[Dict[str,Any]]:
    sql = f"SELECT segment, score FROM `{project}.{dataset}.segment_scores_daily` WHERE date = CURRENT_DATE() ORDER BY score DESC LIMIT 20"
    return [dict(r) for r in bq.query(sql).result()]


def select_ts(segments: List[Dict[str,Any]]) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    best = None
    best_s = -1.0
    ann = []
    for s in segments:
        # Map uplift score into pseudo-success space
        p = max(0.001, min(0.5, float(s.get('score') or 0.0)))
        alpha = 1.0 + 1000.0 * p
        beta = 1.0 + 1000.0 * (1.0 - p)
        sample = float(np.random.beta(alpha, beta))
        row = {'ad_id': s['segment'], 'prior_alpha': alpha, 'prior_beta': beta, 'posterior_alpha': alpha, 'posterior_beta': beta, 'sample': sample}
        ann.append(row)
        if sample > best_s:
            best_s = sample
            best = row
    return best or {}, ann


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    now = datetime.utcnow().isoformat()
    if args.dry_run or not project or not dataset:
        segs = [{'segment':'segA','score':0.12},{'segment':'segB','score':0.06},{'segment':'segC','score':0.03}]
        sel, ann = select_ts(segs)
        print(json.dumps({'selected': sel.get('ad_id'), 'sample': sel.get('sample',0.0)}))
        return
    bq = bigquery.Client(project=project)
    ensure_table(bq, project, dataset)
    segs = fetch_segments(bq, project, dataset)
    if not segs:
        print('no segments')
        return
    sel, ann = select_ts(segs)
    row = {
        'timestamp': now,
        'platform': 'audience',
        'channel': 'segments',
        'campaign_id': 'audience_bandit',
        'ad_id': sel.get('ad_id'),
        'prior_alpha': sel.get('prior_alpha'),
        'prior_beta': sel.get('prior_beta'),
        'posterior_alpha': sel.get('posterior_alpha'),
        'posterior_beta': sel.get('posterior_beta'),
        'sample': sel.get('sample'),
        'context': json.dumps({'arms': ann[:5]}),
    }
    bq.insert_rows_json(f"{project}.{dataset}.bandit_decisions", [row])
    print('audience bandit logged')


if __name__ == '__main__':
    main()

