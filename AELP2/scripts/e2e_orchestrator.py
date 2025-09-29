#!/usr/bin/env python3
"""
End-to-End Orchestrator for AELP2

Runs the full stack in a smart, gated sequence with clear output and
actionable diagnostics. Defaults to dry-run safe modes where applicable.

Steps (in order):
  1) Preflight ensures + views
  2) Great Expectations gate (dry or BQ mode)
  3) Journeys/Uplift/LTV priors
  4) MMM v1 and LWMMM (dry or real), optional Robyn validator
  5) Channel Attribution (container if allowed)
  6) Bandit service + orchestrator (shadow)
  7) Budget orchestrator (shadow)
  8) Value uploads (dry unless flags enable live)
  9) Optional: build dashboard (Next.js)

Usage examples:
  python3 AELP2/scripts/e2e_orchestrator.py --mode smoke
  python3 AELP2/scripts/e2e_orchestrator.py --mode full --gx_bq --with_containers
  python3 AELP2/scripts/e2e_orchestrator.py --exclude dashboard

Env required:
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Flags impacting behavior:
  AELP2_GX_USE_BQ=1 to use BQ-backed GE suites
  AELP2_CA_ALLOW_RUN=1 / AELP2_ROBYN_ALLOW_RUN=1 when Docker installed
  AELP2_ALLOW_GOOGLE_MUTATIONS, AELP2_ALLOW_BANDIT_MUTATIONS, AELP2_ALLOW_VALUE_UPLOADS for live calls
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List


def _run(cmd: List[str], env=None) -> int:
    print(f"[e2e] run: {' '.join(cmd)}")
    p = subprocess.run(cmd, check=False, env=env)
    return p.returncode


def _bq_log(flow: str, rc_map: Dict[str, int]) -> None:
    try:
        from google.cloud import bigquery  # type: ignore
        project = os.getenv('GOOGLE_CLOUD_PROJECT')
        dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
        if not (project and dataset):
            return
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
        failures = [(k, v) for k, v in rc_map.items() if v != 0]
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'flow': flow,
            'rc_map': json.dumps(rc_map),
            'failures': json.dumps(failures),
            'ok': len(failures) == 0,
        }
        bq.insert_rows_json(table_id, [row])
    except Exception as e:
        print(f"[e2e] bq log failed: {e}")


def _load_env_file(path: str, env: Dict[str,str]) -> None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('export '):
                    line=line[len('export '):]
                if '=' not in line:
                    continue
                k,v=line.split('=',1)
                k=k.strip(); v=v.strip().strip('"').strip("'")
                if k and v and k not in env:
                    env[k]=v
    except FileNotFoundError:
        pass


def _auto_env(env: Dict[str,str]) -> Dict[str,str]:
    # Load .env first, then gmail Ads creds overlay, without printing values
    _load_env_file(os.path.join(os.getcwd(), '.env'), env)
    _load_env_file(os.path.join(os.getcwd(), 'AELP2', 'config', '.google_ads_credentials.gmail.env'), env)
    # Prefer Gmail OAuth client when present
    if all(env.get(k) for k in ['GMAIL_CLIENT_ID','GMAIL_CLIENT_SECRET','GMAIL_REFRESH_TOKEN']):
        env['GOOGLE_ADS_CLIENT_ID'] = env['GMAIL_CLIENT_ID']
        env['GOOGLE_ADS_CLIENT_SECRET'] = env['GMAIL_CLIENT_SECRET']
        env['GOOGLE_ADS_REFRESH_TOKEN'] = env['GMAIL_REFRESH_TOKEN']
        if env.get('GMAIL_CUSTOMER_ID'):
            env['GOOGLE_ADS_CUSTOMER_ID'] = env['GMAIL_CUSTOMER_ID']
    # GA4 service account JSON
    if env.get('GA4_SERVICE_ACCOUNT_JSON') and os.path.isfile(env['GA4_SERVICE_ACCOUNT_JSON']):
        env['GOOGLE_APPLICATION_CREDENTIALS'] = env['GA4_SERVICE_ACCOUNT_JSON']
    # Use BQ backed GE
    env.setdefault('AELP2_GX_USE_BQ','1')
    # Set reasonable tolerance for click<=impressions gate to avoid known reporting artifacts
    env.setdefault('AELP2_GX_CLICK_IMP_TOLERANCE','0.05')
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke','full'], default='smoke', help='smoke (safe/dry) or full (BQ + optional containers)')
    ap.add_argument('--gx_bq', action='store_true', help='Run GE suites against BigQuery (sets AELP2_GX_USE_BQ=1)')
    ap.add_argument('--with_containers', action='store_true', help='Attempt containerized jobs (Robyn/ChannelAttribution) if docker present')
    ap.add_argument('--include', nargs='*', default=[], help='Only run these steps (names: preflight,gx,journeys,mmm,ca,bandits,budget,uploads,dashboard)')
    ap.add_argument('--exclude', nargs='*', default=[], help='Skip these steps')
    ap.add_argument('--auto', action='store_true', help='Auto-detect creds from .env + AELP2/config/*.env and choose safest full run')
    args = ap.parse_args()

    env = os.environ.copy()
    if args.auto:
        env = _auto_env(env)
        # Choose defaults for a one-shot end-to-end run
        args.mode = 'full'
        args.gx_bq = True
        # Use containers only if docker exists and flags are already set
        args.with_containers = bool(shutil.which('docker')) and (env.get('AELP2_ROBYN_ALLOW_RUN')=='1' or env.get('AELP2_CA_ALLOW_RUN')=='1')

    required = ['GOOGLE_CLOUD_PROJECT','BIGQUERY_TRAINING_DATASET']
    miss = [k for k in required if not os.getenv(k)]
    # If auto loaded env contains values, adopt them into the process for downstream libs
    if args.auto and miss:
        for k in required:
            if env.get(k):
                os.environ[k]=env[k]
        miss = [k for k in required if not os.getenv(k)]
    if miss:
        print(f"[e2e] Missing required env: {miss}")
        sys.exit(2)

    steps = ['preflight','gx','journeys','mmm','ca','bandits','budget','uploads','dashboard']
    if args.include:
        steps = [s for s in steps if s in args.include]
    if args.exclude:
        steps = [s for s in steps if s not in args.exclude]

    dry = (args.mode == 'smoke')
    if args.gx_bq:
        env['AELP2_GX_USE_BQ'] = '1'

    rc_map: Dict[str, int] = {}

    # 1) Preflight
    if 'preflight' in steps:
        rc_map['preflight'] = _run([sys.executable, 'AELP2/scripts/preflight.py'], env=env)

    # 2) GE gate
    if 'gx' in steps:
        rc_map['gx'] = _run([sys.executable, 'AELP2/ops/gx/run_checks.py'] + (["--dry_run"] if dry and not args.gx_bq else []), env=env)
        if rc_map['gx'] != 0:
            print('[e2e] GE gate failed; skipping downstream. See console above.')
            _bq_log('aelp2-e2e', rc_map)
            sys.exit(1)

    # 3) Journeys/Uplift/LTV
    if 'journeys' in steps:
        rc_map['journey_paths'] = _run([sys.executable, '-m', 'AELP2.pipelines.journey_path_summary'], env=env)
        rc_map['uplift'] = _run([sys.executable, '-m', 'AELP2.pipelines.propensity_uplift'], env=env)
        rc_map['ltv_priors'] = _run([sys.executable, '-m', 'AELP2.pipelines.ltv_priors'], env=env)

    # 4) MMM + LWMMM + Robyn
    if 'mmm' in steps:
        rc_map['mmm_v1'] = _run([sys.executable, '-m', 'AELP2.pipelines.mmm_service'], env=env)
        rc_map['lwmmm'] = _run([sys.executable, '-m', 'AELP2.pipelines.mmm_lightweightmmm'] + (["--dry_run"] if dry else []), env=env)
        if args.with_containers and shutil.which('docker') and env.get('AELP2_ROBYN_ALLOW_RUN','0') == '1':
            rc_map['robyn'] = _run([sys.executable, '-m', 'AELP2.pipelines.robyn_validator'], env=env)
        else:
            rc_map['robyn'] = _run([sys.executable, '-m', 'AELP2.pipelines.robyn_validator', '--dry_run'], env=env)

    # 5) Channel Attribution
    if 'ca' in steps:
        if args.with_containers and shutil.which('docker') and env.get('AELP2_CA_ALLOW_RUN','0') == '1':
            rc_map['channel_attrib'] = _run([sys.executable, '-m', 'AELP2.pipelines.channel_attribution_r'], env=env)
        else:
            rc_map['channel_attrib'] = _run([sys.executable, '-m', 'AELP2.pipelines.channel_attribution_r', '--dry_run'], env=env)

    # 6) Bandits (shadow)
    if 'bandits' in steps:
        rc_map['bandit_service'] = _run([sys.executable, '-m', 'AELP2.core.optimization.bandit_service'] + (["--dry_run"] if dry else []), env=env)
        rc_map['bandit_orchestrator'] = _run([sys.executable, '-m', 'AELP2.core.optimization.bandit_orchestrator'] + (["--dry_run_json", '{"decisions":[]}'] if dry else []), env=env)

    # 7) Budget orchestrator (shadow)
    if 'budget' in steps:
        rc_map['budget_orchestrator'] = _run([sys.executable, '-m', 'AELP2.core.optimization.budget_orchestrator', '--days', '14', '--top_n', '1'], env=env)

    # 8) Value uploads (dry/live by flags)
    if 'uploads' in steps:
        rc_map['upload_google'] = _run([sys.executable, '-m', 'AELP2.pipelines.upload_google_offline_conversions'], env=env)
        rc_map['upload_meta'] = _run([sys.executable, '-m', 'AELP2.pipelines.upload_meta_capi_conversions'], env=env)

    # 9) Dashboard build
    if 'dashboard' in steps:
        dash_dir = os.getenv('AELP2_DASHBOARD_DIR', 'AELP2/apps/dashboard')
        if not os.path.isdir(dash_dir):
            print(f"[e2e] Dashboard dir not found, skipping: {dash_dir}")
            rc_map['dashboard_npm'] = 0
            rc_map['dashboard_build'] = 0
        else:
            # Install and build inside dashboard directory
            npm_env = (env | {'CI': '1', 'HUSKY': '0'})
            rc_map['dashboard_npm'] = _run(['/bin/bash','-lc', f"cd '{dash_dir}' && npm install --no-audit --no-fund --include=dev"], env=npm_env)
            if rc_map['dashboard_npm'] == 0:
                rc_map['dashboard_build'] = _run(['/bin/bash','-lc', f"cd '{dash_dir}' && npm run build"], env=env)
            else:
                rc_map['dashboard_build'] = 1

    _bq_log('aelp2-e2e', rc_map)
    failures = {k:v for k,v in rc_map.items() if v != 0}
    if failures:
        print('[e2e] Failures:', failures)
        sys.exit(1)
    print('[e2e] All steps completed successfully.')


if __name__ == '__main__':
    main()
