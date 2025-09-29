#!/usr/bin/env python3
"""
Pre-flight environment and tooling checks for AELP2.

Performs:
- Print detected env vars (redacted), tool versions
- Verify/ensure BigQuery dataset and required tables/views
- Attempt ingestion connectivity checks (dry/health checks)

Env required:
- GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import json
import shutil
import subprocess
from datetime import datetime


REQUIRED_ENVS = [
    'GOOGLE_CLOUD_PROJECT',
    'BIGQUERY_TRAINING_DATASET',
]


def _print_header(title: str):
    print(f"\n=== {title} ===")


def _redact(k: str, v: str) -> str:
    secret_keys = (
        'TOKEN', 'SECRET', 'PASSWORD', 'KEY', 'CLIENT_ID', 'CLIENT_SECRET', 'REFRESH', 'WEBHOOK'
    )
    if any(s in k.upper() for s in secret_keys):
        if not v:
            return ''
        if len(v) <= 8:
            return '****'
        return v[:4] + '****' + v[-2:]
    return v


def tool_versions():
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        except Exception as e:
            return f"n/a ({e})"
    print('python3:', _run(['python3', '--version']))
    print('pip:', _run(['python3', '-m', 'pip', '--version']))
    print('node:', _run(['node', '--version']))
    print('npm:', _run(['npm', '--version']))
    print('docker:', _run(['docker', '--version']))


def ensure_bq_contract(project: str, dataset: str):
    # Ensure local package import works when invoked as a script
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as e:  # pragma: no cover
        print(f"BigQuery client unavailable: {e}")
        return
    from AELP2.core.ingestion.bq_loader import ensure_dataset, ensure_table
    bq = bigquery.Client(project=project)
    ds_id = f"{project}.{dataset}"
    ensure_dataset(bq, ds_id)
    # Table/View Contract from docs
    tables = [
        'training_episodes', 'safety_events', 'ab_experiments', 'ab_exposures', 'fidelity_evaluations',
        'canary_changes', 'canary_budgets_snapshot', 'bandit_decisions', 'bandit_change_proposals',
        'bandit_change_approvals', 'platform_skeletons', 'journey_paths_daily', 'segment_scores_daily',
        'segment_audience_map', 'value_uploads_log', 'ops_flow_runs', 'ops_alerts', 'channel_attribution_weekly',
        'youtube_reach_estimates', 'value_uploads_staging', 'mmm_validation', 'ltv_priors_daily'
    ]
    schemas = {}
    from google.cloud import bigquery as bqmod
    # Minimal schemas sufficient for insertion in this repo; real pipelines may evolve schemas.
    schemas['training_episodes'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('episode_id', 'STRING'),
        bqmod.SchemaField('steps', 'INT64'),
        bqmod.SchemaField('auctions', 'INT64'),
        bqmod.SchemaField('wins', 'INT64'),
        bqmod.SchemaField('spend', 'FLOAT'),
        bqmod.SchemaField('revenue', 'FLOAT'),
        bqmod.SchemaField('conversions', 'FLOAT'),
        bqmod.SchemaField('win_rate', 'FLOAT'),
    ]
    schemas['safety_events'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('action', 'STRING'),
        bqmod.SchemaField('actor', 'STRING'),
        bqmod.SchemaField('decision', 'STRING'),
        bqmod.SchemaField('notes', 'STRING'),
    ]
    schemas['ab_experiments'] = [
        bqmod.SchemaField('start', 'TIMESTAMP'),
        bqmod.SchemaField('end', 'TIMESTAMP'),
        bqmod.SchemaField('experiment_id', 'STRING'),
        bqmod.SchemaField('platform', 'STRING'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('status', 'STRING'),
        bqmod.SchemaField('variants', 'JSON'),
    ]
    schemas['ab_exposures'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('experiment_id', 'STRING'),
        bqmod.SchemaField('user_id', 'STRING'),
        bqmod.SchemaField('variant_id', 'STRING'),
        bqmod.SchemaField('context', 'JSON'),
    ]
    schemas['fidelity_evaluations'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('eval', 'STRING'),
        bqmod.SchemaField('score', 'FLOAT'),
        bqmod.SchemaField('details', 'JSON'),
    ]
    schemas['canary_changes'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('run_id', 'STRING'),
        bqmod.SchemaField('customer_id', 'STRING'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('campaign_name', 'STRING'),
        bqmod.SchemaField('old_budget', 'FLOAT'),
        bqmod.SchemaField('new_budget', 'FLOAT'),
        bqmod.SchemaField('delta_pct', 'FLOAT'),
        bqmod.SchemaField('direction', 'STRING'),
        bqmod.SchemaField('shadow', 'BOOL'),
        bqmod.SchemaField('applied', 'BOOL'),
        bqmod.SchemaField('error', 'STRING'),
        bqmod.SchemaField('actor', 'STRING'),
        bqmod.SchemaField('notes', 'STRING'),
    ]
    schemas['canary_budgets_snapshot'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('customer_id', 'STRING'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('campaign_name', 'STRING'),
        bqmod.SchemaField('budget', 'FLOAT'),
        bqmod.SchemaField('amount_micros', 'INT64'),
    ]
    schemas['bandit_decisions'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('platform', 'STRING'),
        bqmod.SchemaField('channel', 'STRING'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('ad_id', 'STRING'),
        bqmod.SchemaField('prior_alpha', 'FLOAT'),
        bqmod.SchemaField('prior_beta', 'FLOAT'),
        bqmod.SchemaField('posterior_alpha', 'FLOAT'),
        bqmod.SchemaField('posterior_beta', 'FLOAT'),
        bqmod.SchemaField('sample', 'FLOAT'),
        bqmod.SchemaField('context', 'JSON'),
    ]
    schemas['bandit_change_proposals'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('platform', 'STRING'),
        bqmod.SchemaField('channel', 'STRING'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('ad_id', 'STRING'),
        bqmod.SchemaField('action', 'STRING'),
        bqmod.SchemaField('exploration_pct', 'FLOAT'),
        bqmod.SchemaField('expected_cac_cap', 'FLOAT'),
        bqmod.SchemaField('reason', 'STRING'),
        bqmod.SchemaField('shadow', 'BOOL'),
        bqmod.SchemaField('applied', 'BOOL'),
        bqmod.SchemaField('context', 'JSON'),
    ]
    schemas['bandit_change_approvals'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('campaign_id', 'STRING'),
        bqmod.SchemaField('ad_id', 'STRING'),
        bqmod.SchemaField('action', 'STRING'),
        bqmod.SchemaField('approved', 'BOOL'),
        bqmod.SchemaField('approver', 'STRING'),
        bqmod.SchemaField('reason', 'STRING'),
    ]
    schemas['platform_skeletons'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('platform', 'STRING'),
        bqmod.SchemaField('campaign_name', 'STRING'),
        bqmod.SchemaField('objective', 'STRING'),
        bqmod.SchemaField('daily_budget', 'FLOAT'),
        bqmod.SchemaField('status', 'STRING'),
        bqmod.SchemaField('notes', 'STRING'),
        bqmod.SchemaField('utm', 'JSON'),
    ]
    schemas['journey_paths_daily'] = [
        bqmod.SchemaField('date', 'DATE'),
        bqmod.SchemaField('path', 'STRING'),
        bqmod.SchemaField('count', 'INT64'),
        bqmod.SchemaField('transition_prob', 'FLOAT'),
    ]
    schemas['segment_scores_daily'] = [
        bqmod.SchemaField('date', 'DATE'),
        bqmod.SchemaField('segment', 'STRING'),
        bqmod.SchemaField('score', 'FLOAT'),
        bqmod.SchemaField('method', 'STRING'),
        bqmod.SchemaField('notes', 'STRING'),
        bqmod.SchemaField('metadata', 'JSON'),
    ]
    schemas['segment_audience_map'] = [
        bqmod.SchemaField('segment', 'STRING'),
        bqmod.SchemaField('audience_id', 'STRING'),
        bqmod.SchemaField('platform', 'STRING'),
    ]
    schemas['value_uploads_log'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('platform', 'STRING'),
        bqmod.SchemaField('payload_ref', 'STRING'),
        bqmod.SchemaField('rows', 'INT64'),
        bqmod.SchemaField('allow_uploads', 'BOOL'),
        bqmod.SchemaField('notes', 'STRING'),
    ]
    schemas['ops_flow_runs'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('flow', 'STRING'),
        bqmod.SchemaField('rc_map', 'JSON'),
        bqmod.SchemaField('failures', 'JSON'),
        bqmod.SchemaField('ok', 'BOOL'),
    ]
    schemas['ops_alerts'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('alert', 'STRING'),
        bqmod.SchemaField('severity', 'STRING'),
        bqmod.SchemaField('details', 'JSON'),
    ]
    schemas['channel_attribution_weekly'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('week_start', 'DATE'),
        bqmod.SchemaField('week_end', 'DATE'),
        bqmod.SchemaField('method', 'STRING'),
        bqmod.SchemaField('channel', 'STRING'),
        bqmod.SchemaField('share', 'FLOAT'),
        bqmod.SchemaField('diagnostics', 'JSON'),
    ]
    schemas['youtube_reach_estimates'] = [
        bqmod.SchemaField('date', 'DATE'),
        bqmod.SchemaField('inventory', 'STRING'),
        bqmod.SchemaField('est_impressions', 'INT64'),
        bqmod.SchemaField('est_unique_reach', 'INT64'),
        bqmod.SchemaField('cpm', 'FLOAT'),
        bqmod.SchemaField('notes', 'STRING'),
    ]
    schemas['value_uploads_staging'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('email', 'STRING'),
        bqmod.SchemaField('phone', 'STRING'),
        bqmod.SchemaField('first_name', 'STRING'),
        bqmod.SchemaField('last_name', 'STRING'),
        bqmod.SchemaField('country', 'STRING'),
        bqmod.SchemaField('postal_code', 'STRING'),
        bqmod.SchemaField('value', 'FLOAT'),
        bqmod.SchemaField('currency', 'STRING'),
        bqmod.SchemaField('conversion_action', 'STRING'),
        bqmod.SchemaField('order_id', 'STRING'),
    ]
    schemas['mmm_validation'] = [
        bqmod.SchemaField('timestamp', 'TIMESTAMP'),
        bqmod.SchemaField('method', 'STRING'),
        bqmod.SchemaField('channel', 'STRING'),
        bqmod.SchemaField('window_start', 'DATE'),
        bqmod.SchemaField('window_end', 'DATE'),
        bqmod.SchemaField('diagnostics', 'JSON'),
    ]
    schemas['ltv_priors_daily'] = [
        bqmod.SchemaField('date', 'DATE'),
        bqmod.SchemaField('segment', 'STRING'),
        bqmod.SchemaField('ltv_30', 'FLOAT'),
        bqmod.SchemaField('ltv_90', 'FLOAT'),
        bqmod.SchemaField('method', 'STRING'),
        bqmod.SchemaField('metadata', 'JSON'),
    ]
    # Ensure tables idempotently
    for t in tables:
        try:
            ensure_table(bq, f"{ds_id}.{t}", schemas.get(t, []),
                         partition_field='timestamp' if any(x in t for x in ['changes', 'approvals', 'runs', 'alerts']) else ('date' if 'daily' in t or t.endswith('estimates') else None))
        except Exception as e:
            print(f"warn: ensure {t} failed: {e}")
    # Views via existing helper
    try:
        subprocess.run(['python3', '-m', 'AELP2.pipelines.create_bq_views', '--project', project, '--dataset', dataset], check=False)
    except Exception as e:
        print(f"warn: create_bq_views failed: {e}")


def ingestion_health():
    # Try running GA4 permissions check and recommendations scanner in dry modes if available
    cmds = [
        ['python3', '-m', 'AELP2.pipelines.ga4_permissions_check'],
        ['python3', '-m', 'AELP2.pipelines.google_recommendations_scanner'],
    ]
    for cmd in cmds:
        try:
            print('[preflight] run:', ' '.join(cmd))
            subprocess.run(cmd, check=False)
        except Exception as e:
            print(f"warn: {cmd} failed: {e}")


def main():
    _print_header('ENV VARS (redacted)')
    for k in sorted(set(REQUIRED_ENVS + [x for x in os.environ.keys() if x.startswith('AELP2_') or x.startswith('GOOGLE_') or x.startswith('GA4_')])):
        print(f"{k}={_redact(k, os.environ.get(k, ''))}")

    _print_header('TOOL VERSIONS')
    tool_versions()

    _print_header('BQ CONTRACT ENSURE')
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if project and dataset:
        ensure_bq_contract(project, dataset)
    else:
        print('Skipping: missing GOOGLE_CLOUD_PROJECT/BIGQUERY_TRAINING_DATASET')

    _print_header('INGESTION HEALTH CHECKS')
    ingestion_health()

    print(f"\n[preflight] Completed at {datetime.utcnow().isoformat()}Z")


if __name__ == '__main__':
    main()
