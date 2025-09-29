#!/usr/bin/env python3
"""
Policy Enforcer (platform-agnostic, stub): evaluates high-level policies and records decisions.

Writes `<project>.<dataset>.policy_enforcement` with the evaluated rule and target.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.policy_enforcement"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('rule', 'STRING'),
            bigquery.SchemaField('target', 'STRING'),
            bigquery.SchemaField('decision', 'STRING'),
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
    # Load optional JSON policy config from env path or string
    import json
    cfg = {}
    path = os.getenv('AELP2_POLICY_CONFIG_PATH')
    if path and os.path.exists(path):
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    else:
        js = os.getenv('AELP2_POLICY_CONFIG_JSON')
        if js:
            try:
                cfg = json.loads(js)
            except Exception:
                cfg = {}

    def write(rule, target, decision, notes):
        bq.insert_rows_json(table_id, [{
            'timestamp': datetime.utcnow().isoformat(),
            'rule': rule,
            'target': target,
            'decision': decision,
            'notes': notes,
        }])

    # Default policy: deny live bids without HITL
    allow_live_bids = bool(cfg.get('allow_live_bids', False))
    require_hitl = bool(cfg.get('require_hitl_for_live_bids', True))
    decision = 'deny' if (require_hitl and not allow_live_bids) else 'allow'
    write('no_live_bids_without_hitl', 'all_platforms', decision, 'Eval from policy config')

    # Example: max daily canary change cap
    max_daily_delta = float(cfg.get('max_canary_daily_delta_pct', 0.10))
    write('max_canary_daily_delta_pct', 'google_ads', 'enforce', f'cap={max_daily_delta}')
    print(f'Policy enforcer wrote decision to {table_id}')


if __name__ == '__main__':
    main()
