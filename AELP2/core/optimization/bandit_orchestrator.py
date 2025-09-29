#!/usr/bin/env python3
"""
Bandit Orchestrator (shadow + HITL)

Reads recent `bandit_decisions` and translates them into executable creative change
proposals (enable/pause/adjust exploration split) under global guardrails:

- Exploration budget cap: default 10% (env `AELP2_EXPLORATION_PCT`)
- CAC guardrail: default 200 (env `AELP2_CAC_CAP`)
- Shadow-only by default; logs proposals to BigQuery `bandit_change_proposals`.

This does NOT mutate platforms. A separate HITL step is required to apply changes.

Usage:
  GOOGLE_CLOUD_PROJECT=... BIGQUERY_TRAINING_DATASET=... \
  python -m AELP2.core.optimization.bandit_orchestrator --lookback 30 --exploration_pct 0.1

Dry-run (no BQ required):
  python -m AELP2.core.optimization.bandit_orchestrator --dry_run_json '{"decisions":[{"platform":"google_ads","channel":"search","campaign_id":"19665933250","ad_id":"764281630751","sample":0.24218}]}'
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
except Exception:
    bigquery = None  # allow dry runs without BQ installed


def ensure_table(bq: 'bigquery.Client', project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.bandit_change_proposals"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("platform", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("action", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("exploration_pct", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("expected_cac_cap", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("reason", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("shadow", "BOOL", mode="REQUIRED"),
            bigquery.SchemaField("applied", "BOOL", mode="REQUIRED"),
            bigquery.SchemaField("context", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="timestamp")
        bq.create_table(table)
        return table_id


def fetch_recent_decisions(bq: 'bigquery.Client', project: str, dataset: str, lookback_days: int) -> List[Dict[str, Any]]:
    sql = f"""
      SELECT * FROM `{project}.{dataset}.bandit_decisions`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC
    """
    return [dict(r) for r in bq.query(sql).result()]


def maybe_fetch_cac(bq: 'bigquery.Client', project: str, dataset: str, campaign_id: str, lookback_days: int) -> float:
    try:
        sql = f"""
          SELECT SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac
          FROM `{project}.{dataset}.ads_ad_performance`
          WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY) AND CURRENT_DATE()
            AND campaign_id = @cid
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter("cid", "STRING", campaign_id)])
        rows = list(bq.query(sql, job_config=job_config).result())
        return float(rows[0].cac or 0.0) if rows else 0.0
    except Exception:
        return 0.0


def build_proposals(decisions: List[Dict[str, Any]], exploration_pct: float, cac_cap: float,
                    bq: 'bigquery.Client'=None, project: str=None, dataset: str=None, lookback_days: int=30) -> List[Dict[str, Any]]:
    proposals = []
    seen = set()
    for d in decisions:
        platform = d.get('platform') or 'google_ads'
        channel = d.get('channel') or 'search'
        cid = str(d.get('campaign_id'))
        ad_id = str(d.get('ad_id'))
        key = (platform, channel, cid, ad_id)
        if key in seen:
            continue
        seen.add(key)
        # CAC guardrail (campaign-level proxy)
        est_cac = maybe_fetch_cac(bq, project, dataset, cid, lookback_days) if (bq and project and dataset) else 0.0
        if est_cac and est_cac > 0 and est_cac > cac_cap:
            reason = f"campaign_cac {est_cac:.2f} exceeds cap {cac_cap:.2f}; propose pause or zero exploration"
            action = 'pause'
            exp_pct = 0.0
        else:
            reason = 'promote_selected_ad_within_exploration_budget'
            action = 'adjust_split'
            exp_pct = float(exploration_pct)
        proposals.append({
            'timestamp': datetime.utcnow().isoformat(),
            'platform': platform,
            'channel': channel,
            'campaign_id': cid,
            'ad_id': ad_id,
            'action': action,
            'exploration_pct': exp_pct,
            'expected_cac_cap': float(cac_cap),
            'reason': reason,
            'shadow': True,
            'applied': False,
            'context': json.dumps({'decision': d}, default=str)
        })
    return proposals


def write_proposals(bq: 'bigquery.Client', table_id: str, proposals: List[Dict[str, Any]]):
    if not proposals:
        return
    bq.insert_rows_json(table_id, proposals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lookback', type=int, default=30)
    p.add_argument('--exploration_pct', type=float, default=float(os.getenv('AELP2_EXPLORATION_PCT', '0.10')))
    p.add_argument('--cac_cap', type=float, default=float(os.getenv('AELP2_CAC_CAP', '200.0')))
    p.add_argument('--dry_run_json', help='JSON string with {"decisions": [...]} to bypass BQ')
    args = p.parse_args()

    if args.dry_run_json:
        data = json.loads(args.dry_run_json)
        decisions = data.get('decisions', [])
        props = build_proposals(decisions, args.exploration_pct, args.cac_cap)
        print(json.dumps({'proposals': props}, indent=2))
        return

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    if bigquery is None:
        raise RuntimeError('google-cloud-bigquery not available; install for non-dry runs')

    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    decisions = fetch_recent_decisions(bq, project, dataset, args.lookback)
    if not decisions:
        print('No recent bandit_decisions; nothing to propose')
        return
    proposals = build_proposals(decisions, args.exploration_pct, args.cac_cap, bq=bq, project=project, dataset=dataset, lookback_days=args.lookback)
    write_proposals(bq, table_id, proposals)
    print(f"Logged {len(proposals)} bandit change proposals to {table_id} (shadow)")


if __name__ == '__main__':
    main()
