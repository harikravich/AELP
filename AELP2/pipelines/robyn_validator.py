#!/usr/bin/env python3
"""
Robyn weekly validator (containerized R) — runner stub with BQ summary write.

Shadow-only: Does not mutate platforms. Attempts to run an R container only if
`AELP2_ROBYN_ALLOW_RUN=1` and `docker` is available; otherwise writes a summary
row to BigQuery noting that the run is pending/setup required.

Creates/uses table: <project>.<dataset>.mmm_validation (time-partitioned)
Fields:
- timestamp TIMESTAMP
- method STRING ('robyn')
- channel STRING
- window_start DATE
- window_end DATE
- diagnostics JSON (includes status/note, optional metrics)

CLI:
- --start/--end YYYY-MM-DD
- --dry_run (skip docker/local R; write note only)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import date, datetime, timedelta
from typing import Any, Dict

try:
    from google.cloud import bigquery  # type: ignore
    from google.cloud.exceptions import NotFound  # type: ignore
    BQ_AVAILABLE = True
except Exception:
    bigquery = None  # type: ignore
    NotFound = Exception  # type: ignore
    BQ_AVAILABLE = False


def _safe_bq(project: str | None):
    if not BQ_AVAILABLE or not project:
        return None
    try:
        return bigquery.Client(project=project)
    except Exception:
        return None


def _ensure_val_table(bq: "bigquery.Client", project: str, dataset: str) -> str:
    table = f"{project}.{dataset}.mmm_validation"
    try:
        bq.get_table(table)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('method', 'STRING'),
            bigquery.SchemaField('channel', 'STRING'),
            bigquery.SchemaField('window_start', 'DATE'),
            bigquery.SchemaField('window_end', 'DATE'),
            bigquery.SchemaField('diagnostics', 'JSON'),
        ]
        t = bigquery.Table(table, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    return table


def _ensure_comparison_table(bq: "bigquery.Client", project: str, dataset: str) -> str:
    table = f"{project}.{dataset}.robyn_comparison_weekly"
    try:
        bq.get_table(table)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('channel', 'STRING'),
            bigquery.SchemaField('week_start', 'DATE'),
            bigquery.SchemaField('week_end', 'DATE'),
            bigquery.SchemaField('robyn_cac', 'FLOAT'),
            bigquery.SchemaField('v1_cac', 'FLOAT'),
            bigquery.SchemaField('diagnostics', 'JSON'),
        ]
        t = bigquery.Table(table, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    return table


def _write_summary(project: str, dataset: str, start: date, end: date, diagnostics: Dict[str, Any]) -> None:
    bq = _safe_bq(project)
    if bq is None:
        print('[robyn] BigQuery not available; skipping summary write')
        return
    table = _ensure_val_table(bq, project, dataset)
    comp = _ensure_comparison_table(bq, project, dataset)
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'method': 'robyn',
        'channel': 'google_ads',
        'window_start': str(start),
        'window_end': str(end),
        'diagnostics': json.dumps(diagnostics),
    }
    bq.insert_rows_json(table, [row])
    comp_row = {
        'timestamp': row['timestamp'],
        'channel': 'google_ads',
        'week_start': str(start),
        'week_end': str(end),
        'robyn_cac': None,
        'v1_cac': None,
        'diagnostics': row['diagnostics'],
    }
    try:
        bq.insert_rows_json(comp, [comp_row])
    except Exception:
        pass
    print(f"[robyn] Wrote validation summary to {table}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = date.fromisoformat(args.start) if args.start else (end_d - timedelta(days=90))

    allow_run = os.getenv('AELP2_ROBYN_ALLOW_RUN', '0') == '1' and shutil.which('docker') is not None
    diagnostics: Dict[str, Any] = {
        'status': 'skipped',
        'note': 'container stub; set AELP2_ROBYN_ALLOW_RUN=1 and ensure docker to run Robyn',
    }
    if args.dry_run or not allow_run:
        _write_summary(project or '', dataset or '', start_d, end_d, diagnostics)
        return 0

    # Attempt container run (best-effort; still writes summary)
    try:
        import subprocess, sys, json as _json
        img = os.getenv('AELP2_ROBYN_IMAGE', 'ghcr.io/aelp/robyn:latest')
        cmd = [
            'docker', 'run', '--rm',
            '-e', f'GOOGLE_CLOUD_PROJECT={project}',
            '-e', f'BIGQUERY_TRAINING_DATASET={dataset}',
            img,
            'Rscript', '/app/run_robyn.R', str(start_d), str(end_d)
        ]
        print('[robyn] running:', ' '.join(cmd))
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        rc = proc.returncode
        out = proc.stdout.strip() if proc.stdout else ''
        diagnostics = {'status': 'ok' if rc == 0 else 'failed', 'container_image': img, 'rc': rc}
        # If container returns JSON with comparison metrics, write to comparison table
        if rc == 0 and out:
            try:
                data = _json.loads(out)
                robyn_cac = float(data.get('robyn_cac')) if 'robyn_cac' in data else None
                v1_cac = float(data.get('v1_cac')) if 'v1_cac' in data else None
                bq = _safe_bq(project)
                if bq is not None:
                    comp = _ensure_comparison_table(bq, project or '', dataset or '')
                    bq.insert_rows_json(comp, [{
                        'timestamp': datetime.utcnow().isoformat(),
                        'channel': 'google_ads',
                        'week_start': str(start_d),
                        'week_end': str(end_d),
                        'robyn_cac': robyn_cac,
                        'v1_cac': v1_cac,
                        'diagnostics': _json.dumps({'source': 'container_stdout'})
                    }])
                    print(f"[robyn] Wrote comparison row to {comp}")
            except Exception as e:
                diagnostics['parse_error'] = str(e)
    except Exception as e:
        diagnostics = {'status': 'failed', 'error': str(e)}
    _write_summary(project or '', dataset or '', start_d, end_d, diagnostics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
