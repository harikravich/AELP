#!/usr/bin/env python3
"""
Run AELP2 end-to-end and print actionable recommendations from BigQuery.

What it does
- Loads env (.env + optional Gmail Ads mapping) and applies safe defaults
- Runs the e2e orchestrator (full mode) with GE gate
- Summarizes latest MMM allocations, canary (budget) proposals, and bandit suggestions
- Prints next-step hints for going live

Usage examples
  python3 AELP2/scripts/run_recommendations.py --auto
  python3 AELP2/scripts/run_recommendations.py --auto --containers
  python3 AELP2/scripts/run_recommendations.py --auto --live  # be careful; enables live mutations

Env expected
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Flags
  --auto         Load .env and choose a safe full run
  --containers   Allow Robyn/ChannelAttribution if Docker + flags present
  --live         Enable live mutations (AELP2_ALLOW_GOOGLE_MUTATIONS=1, etc.)
  --skip-dashboard  Skip building the dashboard
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import shutil
import subprocess
from datetime import datetime, timedelta


def load_env_file(path: str, env: dict) -> None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line = line[len('export '):]
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and v and k not in env:
                    env[k] = v
    except FileNotFoundError:
        pass


def auto_env(env: dict) -> dict:
    load_env_file(os.path.join(os.getcwd(), '.env'), env)
    # Prefer Gmail Ads oauth client when present
    if all(env.get(k) for k in ['GMAIL_CLIENT_ID','GMAIL_CLIENT_SECRET','GMAIL_REFRESH_TOKEN']):
        env['GOOGLE_ADS_CLIENT_ID'] = env['GMAIL_CLIENT_ID']
        env['GOOGLE_ADS_CLIENT_SECRET'] = env['GMAIL_CLIENT_SECRET']
        env['GOOGLE_ADS_REFRESH_TOKEN'] = env['GMAIL_REFRESH_TOKEN']
        if env.get('GMAIL_CUSTOMER_ID'):
            env['GOOGLE_ADS_CUSTOMER_ID'] = env['GMAIL_CUSTOMER_ID']
            # If login CID not set, use same
            env.setdefault('GOOGLE_ADS_LOGIN_CUSTOMER_ID', env['GMAIL_CUSTOMER_ID'])
    # GA4 SA JSON → ADC
    if env.get('GA4_SERVICE_ACCOUNT_JSON') and os.path.isfile(env['GA4_SERVICE_ACCOUNT_JSON']):
        env['GOOGLE_APPLICATION_CREDENTIALS'] = env['GA4_SERVICE_ACCOUNT_JSON']
    # GE defaults
    env.setdefault('AELP2_GX_USE_BQ', '1')
    env.setdefault('AELP2_GX_CLICK_IMP_TOLERANCE', '0.05')
    env.setdefault('AELP2_GX_MAX_VIOLATION_ROWS_PCT', '0.10')
    # If KPI conversion ids configured, default MMM to use KPI-only conversions
    if env.get('AELP2_KPI_CONVERSION_ACTION_IDS'):
        env.setdefault('AELP2_MMM_USE_KPI', '1')
    return env


def run_orchestrator(env: dict, with_containers: bool, live: bool, skip_dashboard: bool) -> int:
    args = ['python3', 'AELP2/scripts/e2e_orchestrator.py', '--auto']
    if skip_dashboard:
        args += ['--exclude', 'dashboard']
    # Live toggles
    if live:
        env['AELP2_ALLOW_GOOGLE_MUTATIONS'] = '1'
        env['AELP2_ALLOW_VALUE_UPLOADS'] = '1'
        env['AELP2_VALUE_UPLOAD_DRY_RUN'] = '0'
    else:
        env.setdefault('AELP2_ALLOW_GOOGLE_MUTATIONS', '0')
        env.setdefault('AELP2_ALLOW_VALUE_UPLOADS', '0')
        env.setdefault('AELP2_VALUE_UPLOAD_DRY_RUN', '1')
    # Containers only if requested and docker present and flags already set
    if with_containers and shutil.which('docker'):
        env.setdefault('AELP2_ROBYN_ALLOW_RUN', '1')
        env.setdefault('AELP2_CA_ALLOW_RUN', '1')
    print('[recs] running orchestrator:', ' '.join(args))
    return subprocess.run(args, check=False, env=env).returncode


def summarize_recommendations(env: dict) -> dict:
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as e:
        print('[recs] BigQuery client not available:', e)
        return {}
    project = env.get('GOOGLE_CLOUD_PROJECT')
    dataset = env.get('BIGQUERY_TRAINING_DATASET')
    if not (project and dataset):
        print('[recs] Missing GOOGLE_CLOUD_PROJECT/BIGQUERY_TRAINING_DATASET')
        return {}
    bq = bigquery.Client(project=project)
    results: dict = {}

    def q(sql: str):
        return list(bq.query(sql).result())

    # Latest ops run status
    try:
        rows = q(f"""
          SELECT * FROM `{project}.{dataset}.ops_flow_runs`
          ORDER BY timestamp DESC LIMIT 1
        """)
        if rows:
            r = dict(rows[0])
            results['ops'] = {'ok': r.get('ok'), 'failures': r.get('failures'), 'timestamp': r.get('timestamp')}
    except Exception as e:
        print('[recs] ops_flow_runs read failed:', e)

    # MMM allocations (latest window)
    try:
        rows = q(f"""
          SELECT * FROM `{project}.{dataset}.mmm_allocations`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
          ORDER BY timestamp DESC LIMIT 100
        """)
        results['mmm_allocations'] = [dict(r) for r in rows]
    except Exception as e:
        print('[recs] mmm_allocations read failed:', e)

    # Canary proposals (shadow/live)
    try:
        rows = q(f"""
          SELECT * FROM `{project}.{dataset}.canary_changes`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          ORDER BY timestamp DESC LIMIT 100
        """)
        results['canaries'] = [dict(r) for r in rows]
    except Exception as e:
        print('[recs] canary_changes read failed:', e)

    # Bandit change proposals
    try:
        rows = q(f"""
          SELECT * FROM `{project}.{dataset}.bandit_change_proposals`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          ORDER BY timestamp DESC LIMIT 100
        """)
        results['bandit_proposals'] = [dict(r) for r in rows]
    except Exception as e:
        print('[recs] bandit_change_proposals read failed:', e)

    # Value uploads log (dry/live)
    try:
        rows = q(f"""
          SELECT * FROM `{project}.{dataset}.value_uploads_log`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
          ORDER BY timestamp DESC LIMIT 50
        """)
        results['value_uploads'] = [dict(r) for r in rows]
    except Exception as e:
        print('[recs] value_uploads_log read failed:', e)

    return results


def print_summary(results: dict) -> None:
    print('\n=== AELP2 Recommendations Summary ===')
    ops = results.get('ops') or {}
    if ops:
        print(f"Ops run at {ops.get('timestamp')} | ok={ops.get('ok')} | failures={ops.get('failures')}")
    else:
        print('Ops run status unavailable')

    # Budget moves from canaries if present
    canaries = results.get('canaries') or []
    if canaries:
        print('\nTop Budget Proposals (shadow/live):')
        for c in canaries[:5]:
            cid = c.get('campaign_id')
            direction = c.get('direction')
            delta = c.get('delta_pct')
            notes = c.get('notes') or ''
            print(f"- Campaign {cid}: {direction} {delta:+.1f}% {('— '+notes) if notes else ''}")
    else:
        print('\nNo canary proposals found in the last 7 days.')

    # MMM allocations
    allocs = results.get('mmm_allocations') or []
    if allocs:
        print('\nLatest MMM Allocations (sample):')
        for a in allocs[:5]:
            ch = a.get('channel') or a.get('platform') or a.get('campaign_id')
            pdb = a.get('proposed_daily_budget')
            cac = a.get('expected_cac') or a.get('cac')
            print(f"- {ch}: proposed_daily_budget={pdb} | expected_cac={cac}")
    else:
        print('\nNo MMM allocations available (check training data).')

    # Bandit proposals
    bandits = results.get('bandit_proposals') or []
    if bandits:
        print('\nBandit Creative Suggestions (sample):')
        for b in bandits[:5]:
            print(f"- Campaign {b.get('campaign_id')} ad {b.get('ad_id')}: action={b.get('action')} exploration%={b.get('exploration_pct')} reason={b.get('reason')}")
    else:
        print('\nNo bandit proposals found (check bandit_decisions inputs).')

    # Value uploads mode
    vlogs = results.get('value_uploads') or []
    if vlogs:
        dry = [v for v in vlogs if not v.get('allow_uploads')]
        live = [v for v in vlogs if v.get('allow_uploads')]
        print(f"\nValue Uploads: {len(live)} live, {len(dry)} dry in last 7 days.")
    print('\nNext step: enable live with\n  export AELP2_ALLOW_GOOGLE_MUTATIONS=1\n  export AELP2_ALLOW_VALUE_UPLOADS=1 AELP2_VALUE_UPLOAD_DRY_RUN=0\nthen rerun if you want to apply changes.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto', action='store_true', help='Load .env and apply safe defaults')
    ap.add_argument('--containers', action='store_true', help='Allow container jobs if docker + flags present')
    ap.add_argument('--live', action='store_true', help='Enable live mutations (use with care)')
    ap.add_argument('--skip-dashboard', action='store_true', help='Skip building the dashboard')
    args = ap.parse_args()

    env = os.environ.copy()
    if args.auto:
        env = auto_env(env)
    # Ensure required env propagated
    for k in ('GOOGLE_CLOUD_PROJECT','BIGQUERY_TRAINING_DATASET'):
        if env.get(k):
            os.environ[k] = env[k]

    rc = run_orchestrator(env, with_containers=args.containers, live=args.live, skip_dashboard=args.skip_dashboard)
    if rc != 0:
        print(f'[recs] orchestrator exited with rc={rc}, attempting to summarize what is available...')

    results = summarize_recommendations(env)
    print_summary(results)


if __name__ == '__main__':
    main()
