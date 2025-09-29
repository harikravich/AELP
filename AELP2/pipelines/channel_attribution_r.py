#!/usr/bin/env python3
"""
ChannelAttribution (R) weekly job — containerized stub with BQ writer.

Writes weekly attribution summary to <project>.<dataset>.channel_attribution_weekly.
Shadow-only; runs container only when AELP2_CA_ALLOW_RUN=1 and docker is present.
Otherwise, writes a summary row noting stub status.
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


def _ensure_table(bq: "bigquery.Client", project: str, dataset: str) -> str:
    table = f"{project}.{dataset}.channel_attribution_weekly"
    try:
        bq.get_table(table)
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('week_start', 'DATE'),
            bigquery.SchemaField('week_end', 'DATE'),
            bigquery.SchemaField('method', 'STRING'),
            bigquery.SchemaField('channel', 'STRING'),
            bigquery.SchemaField('share', 'FLOAT'),
            bigquery.SchemaField('diagnostics', 'JSON'),
        ]
        t = bigquery.Table(table, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
    return table


def _write_stub(project: str, dataset: str, week: tuple[date, date], diagnostics: Dict[str, Any]) -> None:
    bq = _safe_bq(project)
    if bq is None:
        print('[ca] BigQuery not available; skipping write')
        return
    table = _ensure_table(bq, project, dataset)
    ws, we = week
    # Write a single aggregate row (channel='all')
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'week_start': str(ws),
        'week_end': str(we),
        'method': 'channel_attribution_stub',
        'channel': 'all',
        'share': 1.0,
        'diagnostics': json.dumps(diagnostics),
    }
    bq.insert_rows_json(table, [row])
    print(f"[ca] Wrote weekly attribution stub to {table}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start')
    p.add_argument('--end')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = date.fromisoformat(args.start) if args.start else (end_d - timedelta(days=7))
    allow_run = os.getenv('AELP2_CA_ALLOW_RUN', '0') == '1' and shutil.which('docker') is not None
    diagnostics = {'status': 'skipped', 'note': 'container not run; set AELP2_CA_ALLOW_RUN=1 + docker'}
    if args.dry_run or not allow_run:
        _write_stub(project or '', dataset or '', (start_d, end_d), diagnostics)
        return 0
    try:
        import subprocess, json as _json
        img = os.getenv('AELP2_CA_IMAGE', 'ghcr.io/aelp/channelattribution:latest')
        cmd = ['docker', 'run', '--rm', '-e', f'GOOGLE_CLOUD_PROJECT={project}', '-e', f'BIGQUERY_TRAINING_DATASET={dataset}', img]
        print('[ca] running:', ' '.join(cmd))
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        rc = proc.returncode
        stdout = proc.stdout.strip() if proc.stdout else ''
        diagnostics = {'status': 'ok' if rc == 0 else 'failed', 'rc': rc, 'container_image': img}
        # If container printed JSON rows, parse and write them
        if rc == 0 and stdout:
            try:
                data = _json.loads(stdout)
                rows = data if isinstance(data, list) else data.get('rows', [])
                parsed = []
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    parsed.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'week_start': str(start_d),
                        'week_end': str(end_d),
                        'method': str(r.get('method') or 'markov'),
                        'channel': str(r.get('channel') or 'unknown'),
                        'share': float(r.get('share') or 0.0),
                        'diagnostics': _json.dumps({'source': 'container_stdout'})
                    })
                if parsed:
                    bq = _safe_bq(project)
                    if bq is not None:
                        table = _ensure_table(bq, project or '', dataset or '')
                        bq.insert_rows_json(table, parsed)
                        print(f"[ca] Wrote {len(parsed)} container attribution rows to {table}")
                        return 0
            except Exception as e:
                diagnostics['parse_error'] = str(e)
    except Exception as e:
        diagnostics = {'status': 'failed', 'error': str(e)}
    _write_stub(project or '', dataset or '', (start_d, end_d), diagnostics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
