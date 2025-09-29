#!/usr/bin/env python3
"""
Summarize user journey paths and transition probabilities.

Reads from <project>.gaelp_users.journey_sessions with columns:
- user_id STRING
- session_start TIMESTAMP
- default_channel_group STRING (or channel STRING)

Writes to <project>.<dataset>.journey_paths_daily for current date.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, date
from typing import List, Dict

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.journey_paths_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('path', 'STRING'),
            bigquery.SchemaField('count', 'INT64'),
            bigquery.SchemaField('transition_prob', 'FLOAT'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        return table_id


def fetch_sessions(bq: bigquery.Client, project: str) -> List[Dict]:
    # Use last 30 days for path summaries
    sql = f"""
      SELECT user_id, session_start, COALESCE(default_channel_group, channel) AS ch
      FROM `{project}.gaelp_users.journey_sessions`
      WHERE DATE(session_start) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
    """
    try:
        return [dict(r) for r in bq.query(sql).result()]
    except Exception:
        return []


def compute_paths(rows: List[Dict]) -> List[Dict]:
    by_user: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_user[str(r['user_id'])].append(r)
    path_counts: Counter = Counter()
    trans_counts: Counter = Counter()
    starts: Counter = Counter()
    for uid, sess in by_user.items():
        sess.sort(key=lambda x: x['session_start'])
        channels = [str(x['ch'] or 'unknown') for x in sess]
        if not channels:
            continue
        path = '>'.join(channels[:10])
        path_counts[path] += 1
        # transitions
        starts[channels[0]] += 1
        for i in range(len(channels)-1):
            trans = (channels[i], channels[i+1])
            trans_counts[trans] += 1
    # Convert transitions to probabilities
    trans_prob: Dict[str, float] = {}
    for (a, b), c in trans_counts.items():
        denom = sum(v for (x, _), v in trans_counts.items() if x == a)
        if denom:
            trans_prob[f"{a}>{b}"] = c / denom
    today = date.today().isoformat()
    out = [{'date': today, 'path': k, 'count': int(v), 'transition_prob': None} for k, v in path_counts.items()]
    out += [{'date': today, 'path': k, 'count': None, 'transition_prob': float(p)} for k, p in trans_prob.items()]
    return out


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    rows = fetch_sessions(bq, project)
    if not rows:
        print('No journey sessions found; ensured journey_paths_daily table')
        return
    out = compute_paths(rows)
    if out:
        bq.insert_rows_json(table_id, out)
        print(f"Wrote {len(out)} journey path rows to {table_id}")
    else:
        print('No journey paths computed')


if __name__ == '__main__':
    main()

