#!/usr/bin/env python3
"""
Promote policy hints to shadow proposals (HITL path, stub).

Reads `<project>.<dataset>.policy_hints` and writes shadow proposals into
`<project>.<dataset>.bandit_change_proposals`.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, table_id: str, schema):
    try:
        bq.get_table(table_id)
    except NotFound:
        bq.create_table(bigquery.Table(table_id, schema=schema))


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    hints_tbl = f"{project}.{dataset}.policy_hints"
    props_tbl = f"{project}.{dataset}.bandit_change_proposals"
    # Ensure proposals table (minimal schema)
    ensure_table(bq, props_tbl, [
        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
        bigquery.SchemaField('platform', 'STRING'),
        bigquery.SchemaField('channel', 'STRING'),
        bigquery.SchemaField('campaign_id', 'STRING'),
        bigquery.SchemaField('ad_id', 'STRING'),
        bigquery.SchemaField('action', 'STRING'),
        bigquery.SchemaField('exploration_pct', 'FLOAT'),
        bigquery.SchemaField('reason', 'STRING'),
        bigquery.SchemaField('shadow', 'BOOL'),
        bigquery.SchemaField('applied', 'BOOL'),
    ])
    # Read hints (last 7d)
    try:
        rows = list(bq.query(f"""
          SELECT timestamp, hint_type, target, value
          FROM `{hints_tbl}`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          ORDER BY timestamp DESC LIMIT 50
        """).result())
    except Exception:
        rows = []
    out = []
    now = datetime.utcnow().isoformat()
    for r in rows:
        hint_type = str(r.hint_type)
        if hint_type == 'exploration':
            out.append({
                'timestamp': now,
                'platform': 'google_ads',
                'channel': 'search',
                'campaign_id': 'canary',
                'ad_id': None,
                'action': 'explore',
                'exploration_pct': float(r.value or 0.1),
                'reason': f'policy_hint:{r.target}',
                'shadow': True,
                'applied': False,
            })
        elif hint_type == 'budget_tilt':
            out.append({
                'timestamp': now,
                'platform': 'google_ads',
                'channel': 'search',
                'campaign_id': 'canary',
                'ad_id': None,
                'action': 'budget_tilt',
                'exploration_pct': float(r.value or 0.05),
                'reason': f'policy_hint:{r.target}',
                'shadow': True,
                'applied': False,
            })
    if out:
        bq.insert_rows_json(props_tbl, out)
        print(f"Promoted {len(out)} hints to shadow proposals")
    else:
        print('No hints to promote')


if __name__ == '__main__':
    main()

