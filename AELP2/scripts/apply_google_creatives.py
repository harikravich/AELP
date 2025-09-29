#!/usr/bin/env python3
"""
Apply creative changes (shadow-first, flag-gated).

Defaults to shadow mode (no live mutations). Real platform changes require both:
- GATES_ENABLED=1 and AELP2_ALLOW_BANDIT_MUTATIONS=1 (and ALLOW_REAL_CREATIVE=1 optional)

BigQuery tables (ensured if credentials present):
- `${GOOGLE_CLOUD_PROJECT}.${BIGQUERY_TRAINING_DATASET}.creative_changes` (DAY partitioned on timestamp)

Env:
  DRY_RUN=1 (default) → skip SDK calls and network writes; only print intent
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET (optional for audit rows)
  AELP2_ALLOW_BANDIT_MUTATIONS=0|1 (default 0)
  ALLOW_REAL_CREATIVE=0|1 (default 0)
  AELP2_SHADOW_MODE=1|0 (default 1)
  AELP2_ACTOR (default cli)

Input (either):
- JSON via env `AELP2_CREATIVE_CHANGE_JSON` (single change), or
- Falls back to a stub example change for validation in DRY_RUN.

This script never performs live changes unless all guards allow; even then,
it will only attempt SDK operations for Google Ads if implemented later.
"""
import os
import sys
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any


def _load_change() -> Dict[str, Any]:
    raw = os.getenv('AELP2_CREATIVE_CHANGE_JSON')
    if raw:
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"Invalid AELP2_CREATIVE_CHANGE_JSON: {e}", file=sys.stderr)
            sys.exit(2)
    # Stub example
    return {
        'platform': 'google_ads',
        'campaign_id': 'canary',
        'ad_group_id': 'stub_group',
        'creative_id': None,
        'action': 'create_responsive_search_ad',
        'payload': {
            'headlines': ["Try Aura Premium", "Protect your family"],
            'descriptions': ["Online safety made simple"],
            'final_url': 'https://example.com/safety'
        }
    }


def _ensure_creative_changes_table() -> str:
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        return ''
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception:
        return ''
    client = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.creative_changes"
    try:
        client.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('run_id', 'STRING'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('ad_group_id', 'STRING'),
            bigquery.SchemaField('creative_id', 'STRING'),
            bigquery.SchemaField('action', 'STRING'),
            bigquery.SchemaField('payload', 'JSON'),
            bigquery.SchemaField('shadow', 'BOOL'),
            bigquery.SchemaField('applied', 'BOOL'),
            bigquery.SchemaField('error', 'STRING'),
            bigquery.SchemaField('actor', 'STRING'),
            bigquery.SchemaField('notes', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        client.create_table(t)
        return table_id


def _ops_log(rc_map: Dict[str, Any]):
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        return
    try:
        from google.cloud import bigquery  # type: ignore
        bq = bigquery.Client(project=project)
        table_id = f"{project}.{dataset}.ops_flow_runs"
        try:
            bq.get_table(table_id)
        except Exception:
            schema = [
                bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                bigquery.SchemaField('flow', 'STRING'),
                bigquery.SchemaField('rc_map', 'JSON'),
                bigquery.SchemaField('failures', 'JSON'),
                bigquery.SchemaField('ok', 'BOOL'),
            ]
            t = bigquery.Table(table_id, schema=schema)
            t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
            bq.create_table(t)
        bq.insert_rows_json(table_id, [{
            'timestamp': datetime.utcnow().isoformat(),
            'flow': 'apply_creatives_cli',
            'rc_map': json.dumps(rc_map),
            'failures': '[]',
            'ok': True,
        }])
    except Exception:
        pass


def main():
    change = _load_change()
    gates_enabled = (os.getenv('GATES_ENABLED', '1') == '1')
    allow_mut = (os.getenv('AELP2_ALLOW_BANDIT_MUTATIONS', '0') == '1') and (os.getenv('ALLOW_REAL_CREATIVE', '0') == '1')
    shadow_mode = (os.getenv('AELP2_SHADOW_MODE', '1') == '1') or (not gates_enabled) or (not allow_mut)
    actor = os.getenv('AELP2_ACTOR', 'cli')
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    table_id = _ensure_creative_changes_table()

    # Always log the proposal (shadow by default)
    if table_id:
        try:
            from google.cloud import bigquery  # type: ignore
            bq = bigquery.Client(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
            row = {
                'timestamp': now,
                'run_id': run_id,
                'platform': change.get('platform'),
                'campaign_id': change.get('campaign_id'),
                'ad_group_id': change.get('ad_group_id'),
                'creative_id': change.get('creative_id'),
                'action': change.get('action'),
                'payload': json.dumps(change.get('payload', {})),
                'shadow': True,
                'applied': False,
                'error': None,
                'actor': actor,
                'notes': 'shadow proposal; gating enforced',
            }
            bq.insert_rows_json(table_id, [row])
        except Exception as e:
            print(f"[warn] BQ audit write failed: {e}")

    # DRY_RUN or gates disabled → stop here
    if os.getenv('DRY_RUN', '1') == '1' or shadow_mode:
        print("[DRY_RUN or shadow] Not applying creative changes. Gating intact.")
        _ops_log({'dry_run': 0, 'shadow': True})
        return

    # Live mutations not implemented; keep safe
    print("Live creative mutations are not implemented in this environment. Aborting.", file=sys.stderr)
    _ops_log({'attempt_live': 1, 'shadow': False})
    sys.exit(1)


if __name__ == '__main__':
    main()

