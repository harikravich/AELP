#!/usr/bin/env python3
"""
Bandit v1 (Thompson Sampling, shadow-only) for Creative Selection on Google Ads.

Reads creative arms (ads) for a target campaign from BigQuery `ads_ad_performance`
over a lookback window, constructs Beta posteriors using clicks/impressions (CTR proxy),
draws Thompson samples, selects the best arm, and logs the decision to
`<project>.<dataset>.bandit_decisions` (creating it if missing).

Notes:
- Shadow-only: no platform mutations are made.
- This is a bootstrap implementation using CTR as the reward proxy; we can upgrade to CVR/ROAS later.
- Exploration budget and CAC caps are enforced at the orchestrator level; this service only logs decisions.

Usage:
  python -m AELP2.core.optimization.bandit_service --lookback 30 --campaign_id <id>
  # or let it auto-pick the top campaign by spend in last 30 days

Env required: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
"""

import os
import argparse
import json
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Tuple

import numpy as np
try:
    from mabwiser.mab import MAB, LearningPolicy
    HAVE_MABWISER = True
except Exception:
    HAVE_MABWISER = False
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_bandit_table(bq: bigquery.Client, project: str, dataset: str) -> None:
    table_id = f"{project}.{dataset}.bandit_decisions"
    try:
        bq.get_table(table_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("platform", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("prior_alpha", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("prior_beta", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("posterior_alpha", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("posterior_beta", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("sample", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("context", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="timestamp"
        )
        bq.create_table(table)


def ensure_ab_experiments(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.ab_experiments"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('start', 'TIMESTAMP'),
            bigquery.SchemaField('end', 'TIMESTAMP'),
            bigquery.SchemaField('experiment_id', 'STRING'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('status', 'STRING'),
            bigquery.SchemaField('variants', 'JSON'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='start')
        bq.create_table(t)
        return table_id


def pick_top_campaign(bq: bigquery.Client, project: str, dataset: str, lookback: int) -> str:
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS cid, SUM(cost_micros)/1e6 AS spend
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {lookback} DAY) AND CURRENT_DATE()
      GROUP BY cid
      ORDER BY spend DESC
      LIMIT 1
    """
    rows = list(bq.query(sql).result())
    return rows[0].cid if rows else ''


def fetch_creative_arms(bq: bigquery.Client, project: str, dataset: str, campaign_id: str, lookback: int) -> List[Dict[str, Any]]:
    sql = f"""
      SELECT ad_id, impressions, clicks FROM (
        SELECT CAST(ad_id AS STRING) AS ad_id,
               SUM(CAST(impressions AS INT64)) AS impressions,
               SUM(CAST(clicks AS INT64)) AS clicks
        FROM `{project}.{dataset}.ads_ad_performance`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {lookback} DAY) AND CURRENT_DATE()
          AND campaign_id = '{campaign_id}'
        GROUP BY ad_id
      )
      WHERE impressions > 0
      ORDER BY impressions DESC
    """
    return [dict(r) for r in bq.query(sql).result()]


def thompson_select(arms: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute Beta priors and draw samples; return selected arm and annotated arms."""
    annotated = []
    best = None
    best_sample = -1.0
    for arm in arms:
        imps = float(arm['impressions'] or 0.0)
        clicks = float(arm['clicks'] or 0.0)
        # Beta(1+clicks, 1+(imps-clicks)) as CTR prior
        alpha = 1.0 + max(0.0, clicks)
        beta = 1.0 + max(0.0, imps - clicks)
        sample = np.random.beta(alpha, beta)
        row = {
            **arm,
            'prior_alpha': alpha,
            'prior_beta': beta,
            'posterior_alpha': alpha,  # same in bootstrap
            'posterior_beta': beta,
            'sample': float(sample),
        }
        annotated.append(row)
        if sample > best_sample:
            best_sample = sample
            best = row
    return best or {}, annotated


def mabwiser_select(arms: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Use MABWiser TS if available. Fits on aggregated clicks/impressions as Bernoulli.
    We replicate arm stats as weighted observations by providing per-arm successes/failures.
    """
    if not HAVE_MABWISER:
        return {}, []
    # Prepare data
    labels = [str(a['ad_id']) for a in arms]
    pulls = []
    rewards = []
    for a in arms:
        imps = int(a.get('impressions') or 0)
        clicks = int(a.get('clicks') or 0)
        fails = max(0, imps - clicks)
        # Add successes and failures as weighted samples
        pulls.extend([str(a['ad_id'])] * (clicks + fails))
        rewards.extend([1] * clicks + [0] * fails)
    if not pulls:
        return {}, []
    mab = MAB(arms=labels, learning_policy=LearningPolicy.ThompsonSampling())
    mab.fit(decisions=pulls, rewards=rewards)
    # Draw a recommendation
    rec = mab.predict()
    annotated = []
    selected = None
    for a in arms:
        ad = str(a['ad_id'])
        # MABWiser does not expose posterior params; store mean estimate
        est = mab.predict_expectations()[ad]
        row = {**a, 'sample': float(est), 'posterior_alpha': float(a.get('clicks', 0)) + 1.0, 'posterior_beta': float((a.get('impressions', 0) or 0) - (a.get('clicks', 0) or 0)) + 1.0}
        annotated.append(row)
        if ad == rec:
            selected = row
    return selected or {}, annotated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lookback', type=int, default=30)
    p.add_argument('--campaign_id', help='Google Ads campaign id (string)')
    p.add_argument('--dry_run', action='store_true', help='Run locally with synthetic arms; no BQ')
    args = p.parse_args()

    if args.dry_run:
        rng = np.random.default_rng(0)
        arms = []
        for i in range(6):
            imps = int(rng.integers(1000, 10000))
            ctr = rng.uniform(0.01, 0.08)
            clicks = int(imps * ctr)
            arms.append({'ad_id': f'dry_{i}', 'impressions': imps, 'clicks': clicks})
        campaign_id = 'dry_run_campaign'
    else:
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
        if not project or not dataset:
            raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
        bq = bigquery.Client(project=project)
        ensure_bandit_table(bq, project, dataset)
        campaign_id = args.campaign_id or pick_top_campaign(bq, project, dataset, args.lookback)
        if not campaign_id:
            print('No campaign found; aborting')
            return
        arms = fetch_creative_arms(bq, project, dataset, campaign_id, args.lookback)
        if not arms:
            print(f'No arms (ads) found for campaign {campaign_id}; aborting')
            return

    # Choose engine
    engine = os.getenv('AELP2_BANDIT_ENGINE', 'auto')
    if engine == 'mabwiser' or (engine == 'auto' and HAVE_MABWISER):
        selected, annotated = mabwiser_select(arms)
        if not selected:
            selected, annotated = thompson_select(arms)
    else:
        selected, annotated = thompson_select(arms)
    now = datetime.utcnow().isoformat()

    if args.dry_run:
        print(json.dumps({'campaign_id': campaign_id, 'selected_ad': selected.get('ad_id'), 'sample': selected.get('sample', 0.0), 'engine': 'mabwiser' if HAVE_MABWISER else 'ts'}))
        return
    # Log selected arm into bandit_decisions
    table_id = f"{project}.{dataset}.bandit_decisions"
    row = {
        'timestamp': now,
        'platform': 'google_ads',
        'channel': 'search',
        'campaign_id': str(campaign_id),
        'ad_id': str(selected.get('ad_id')),
        'prior_alpha': float(selected.get('prior_alpha', 1.0)),
        'prior_beta': float(selected.get('prior_beta', 1.0)),
        'posterior_alpha': float(selected.get('posterior_alpha', 1.0)),
        'posterior_beta': float(selected.get('posterior_beta', 1.0)),
        'sample': float(selected.get('sample', 0.0)),
        'context': json.dumps({'lookback_days': args.lookback, 'arms': annotated[:5]}),
    }
    bq.insert_rows_json(table_id, [row])
    print(f"Logged bandit decision for campaign {campaign_id}: ad {row['ad_id']} (sample={row['sample']:.5f})")
    # Seed AB experiment entry for transparency
    try:
        exp_tbl = ensure_ab_experiments(bq, project, dataset)
        exp_row = {
            'start': now,
            'end': None,
            'experiment_id': f"bandit_seed_{campaign_id}_{row['ad_id']}_{now[:10]}",
            'platform': 'google_ads',
            'campaign_id': str(campaign_id),
            'status': 'proposed',
            'variants': json.dumps([{'variant_id': str(row['ad_id']), 'posterior_mean': float(row.get('posterior_alpha',1.0)) / max(1.0, float(row.get('posterior_alpha',1.0) + row.get('posterior_beta',1.0)))}]),
        }
        bq.insert_rows_json(exp_tbl, [exp_row])
    except Exception:
        pass


if __name__ == '__main__':
    main()
