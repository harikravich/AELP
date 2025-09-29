#!/usr/bin/env python3
"""
Budget Orchestrator (shadow):

Reads the latest MMM allocation for channel 'google_ads', compares to recent spend,
and issues shadow canary budget proposals for top campaigns by recent spend using
the existing apply_google_canary.py (shadow mode). Caps are enforced by that script.

Usage:
  python -m AELP2.core.optimization.budget_orchestrator --days 14 --top_n 1

Env required:
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  GOOGLE_ADS_LOGIN_CUSTOMER_ID, GOOGLE_ADS_CUSTOMER_ID (for apply_google_canary)

Notes:
  - This is a bootstrap orchestrator; it operates on total Google Ads only and
    selects top-N campaigns by recent spend to nudge toward the MMM target.
  - All actions are shadow-only; no live mutations are made.
"""

import os
import json
import argparse
import subprocess
from datetime import date, timedelta
from typing import List, Dict, Tuple

from google.cloud import bigquery


def get_latest_allocation(bq: bigquery.Client, project: str, dataset: str) -> Dict:
    sql = f"""
      SELECT * FROM `{project}.{dataset}.mmm_allocations`
      WHERE channel='google_ads'
      ORDER BY timestamp DESC LIMIT 1
    """
    rows = list(bq.query(sql).result())
    return dict(rows[0]) if rows else {}


def parse_uncertainty(alloc: Dict) -> float:
    try:
        diag = alloc.get('diagnostics')
        if isinstance(diag, str):
            diag = json.loads(diag)
        return float(diag.get('uncertainty_pct', 0.2)) if diag else 0.2
    except Exception:
        return 0.2


def conservative_cac(alloc: Dict, unc: float) -> float:
    try:
        ecac = float(alloc.get('expected_cac') or 0.0)
        if ecac <= 0:
            return 0.0
        # Use lower-bound conversions → higher CAC; approximate by scaling up CAC by uncertainty
        return ecac * (1.0 + max(0.0, float(unc)))
    except Exception:
        return 0.0


def get_recent_spend(bq: bigquery.Client, project: str, dataset: str, days: int) -> float:
    sql = f"""
      SELECT SUM(cost) AS s
      FROM `{project}.{dataset}.ads_campaign_daily`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND CURRENT_DATE()
    """
    rows = list(bq.query(sql).result())
    return float(rows[0].s or 0.0) if rows else 0.0


def estimate_campaign_cac(bq: bigquery.Client, project: str, dataset: str, campaign_id: str, days: int) -> float:
    sql = f"""
      SELECT SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac
      FROM `{project}.{dataset}.ads_ad_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND CURRENT_DATE()
        AND CAST(campaign_id AS STRING) = @cid
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter('cid', 'STRING', str(campaign_id))]
    ))
    rows = list(job.result())
    return float(rows[0].cac or 0.0) if rows else 0.0


def get_hist_daily_cap(avg_daily: float) -> float:
    try:
        cap_pct = float(os.getenv('AELP2_HIST_DAILY_CAP_PCT', '0.25'))
    except Exception:
        cap_pct = 0.25
    return max(1.0, avg_daily * (1.0 + cap_pct))


def get_top_campaign_ids(bq: bigquery.Client, project: str, dataset: str, days: int, top_n: int) -> List[str]:
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS cid, SUM(cost_micros)/1e6 AS spend
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND CURRENT_DATE()
      GROUP BY cid
      ORDER BY spend DESC
      LIMIT {top_n}
    """
    return [r.cid for r in bq.query(sql).result()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--days', type=int, default=14)
    p.add_argument('--top_n', type=int, default=1)
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    # If Gmail Ads OAuth is present, map its customer id into Ads env
    if os.getenv('GMAIL_CUSTOMER_ID') and not os.getenv('GOOGLE_ADS_CUSTOMER_ID'):
        os.environ['GOOGLE_ADS_CUSTOMER_ID'] = os.getenv('GMAIL_CUSTOMER_ID')  # type: ignore
    if os.getenv('GMAIL_CUSTOMER_ID') and not os.getenv('GOOGLE_ADS_LOGIN_CUSTOMER_ID'):
        os.environ['GOOGLE_ADS_LOGIN_CUSTOMER_ID'] = os.getenv('GMAIL_CUSTOMER_ID')  # type: ignore
    # Require Ads env for downstream connector call (even in shadow, we audit via apply script)
    for v in ['GOOGLE_ADS_CUSTOMER_ID']:
        if not os.getenv(v):
            raise RuntimeError(f'Missing env: {v}')

    bq = bigquery.Client(project=project)

    alloc = get_latest_allocation(bq, project, dataset)
    if not alloc:
        print('No MMM allocation found; aborting')
        return
    target_budget = float(alloc.get('proposed_daily_budget') or 0.0)
    avg_daily_spend = get_recent_spend(bq, project, dataset, args.days) / max(args.days, 1)
    hist_cap = get_hist_daily_cap(avg_daily_spend)
    capped_target = min(target_budget, hist_cap)
    unc_pct = parse_uncertainty(alloc)
    cons_cac = conservative_cac(alloc, unc_pct)
    # Determine direction (conservative on uncertainty & CAC)
    direction = 'up' if capped_target > avg_daily_spend else 'down'
    if cons_cac and cons_cac > 0 and os.getenv('AELP2_CAC_CAP'):
        try:
            if cons_cac > float(os.getenv('AELP2_CAC_CAP','0') or '0'):
                direction = 'down'
        except Exception:
            pass

    # Pick campaigns to propose changes
    ids = get_top_campaign_ids(bq, project, dataset, args.days, args.top_n)
    if not ids:
        print('No campaigns found in ads_campaign_performance; aborting')
        return
    ids_csv = ','.join(ids)

    env = os.environ.copy()
    # Prefer Gmail OAuth if present to avoid MCC permission issues
    if env.get('GMAIL_CLIENT_ID') and env.get('GMAIL_REFRESH_TOKEN'):
        env['GOOGLE_ADS_CLIENT_ID'] = env['GMAIL_CLIENT_ID']
        if env.get('GMAIL_CLIENT_SECRET'):
            env['GOOGLE_ADS_CLIENT_SECRET'] = env['GMAIL_CLIENT_SECRET']
        env['GOOGLE_ADS_REFRESH_TOKEN'] = env['GMAIL_REFRESH_TOKEN']
    env['AELP2_GOOGLE_CANARY_CAMPAIGN_IDS'] = ids_csv
    env['AELP2_CANARY_BUDGET_DELTA_PCT'] = env.get('AELP2_CANARY_BUDGET_DELTA_PCT', '0.05')
    env['AELP2_CANARY_MAX_CHANGES_PER_RUN'] = env.get('AELP2_CANARY_MAX_CHANGES_PER_RUN', str(args.top_n))
    env['AELP2_SHADOW_MODE'] = '1'
    env['AELP2_ALLOW_GOOGLE_MUTATIONS'] = '0'
    env['AELP2_CANARY_BUDGET_DIRECTION'] = direction
    # CAC guardrail: if any selected campaign CAC exceeds cap, force down proposal and record reason
    try:
        cac_cap = float(os.getenv('AELP2_CAC_CAP', '0') or '0')
    except Exception:
        cac_cap = 0.0
    cap_reason = None
    cac_estimates: List[Tuple[str, float]] = []
    if cac_cap > 0:
        for cid in ids:
            est_cac = estimate_campaign_cac(bq, project, dataset, cid, args.days)
            cac_estimates.append((cid, est_cac))
            if est_cac and est_cac > cac_cap:
                cap_reason = f"campaign_cac {est_cac:.2f} exceeds cap {cac_cap:.2f}"
        if cap_reason:
            direction = 'down'

    # Pass uncertainty/cap notes to changefeed for transparency
    notes = json.dumps({
        'mmm_target': target_budget,
        'avg_daily_spend': avg_daily_spend,
        'hist_daily_cap': hist_cap,
        'capped_target': capped_target,
        'uncertainty_pct': unc_pct,
        'conservative_cac': cons_cac,
        'cap_reason': cap_reason,
        'cac_estimates': [{ 'campaign_id': c, 'cac': v } for c, v in cac_estimates],
    })
    env['AELP2_PROPOSAL_NOTES'] = notes

    print(f"Proposing shadow canary for {ids_csv} direction={direction} (avg_daily_spend={avg_daily_spend:.2f}, target={target_budget:.2f}, capped_target={capped_target:.2f})")
    # Call apply_google_canary in the same environment (shadow)
    env['PYTHONPATH'] = env.get('PYTHONPATH', '.')
    cmd = ['python3', 'AELP2/scripts/apply_google_canary.py']
    subprocess.run(cmd, check=False, env=env)


if __name__ == '__main__':
    main()
