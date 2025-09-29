#!/usr/bin/env python3
"""
LP Module Runner (background job)
Scans lp_module_runs with status='running', executes a lightweight connector,
writes module_results and marks status='done'. Demo-safe for P0.
"""
import os, json, time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

from google.cloud import bigquery  # type: ignore

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
DATASET = os.environ.get('BIGQUERY_TRAINING_DATASET')

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def run_connector(slug: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if slug == 'insight_preview':
        return {
            'summary_text': 'Late-night activity looks above typical (runner demo).',
            'result_json': { 'bars': [ {'label':'Day','v': 40}, {'label':'Night','v': 85} ], 'hints': ['Night ~2.1x Day'] },
        }
    if slug == 'scam_check':
        return {
            'summary_text': 'Link risk: Low (runner demo).',
            'result_json': { 'risk':'low', 'reasons': ['Domain age > 6 months','TLS valid'] },
        }
    return { 'summary_text': 'Module not recognized', 'result_json': {} }

def main():
    if not PROJECT or not DATASET:
        print('[runner] Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET')
        return 2
    bq = bigquery.Client(project=PROJECT)
    # ensure tables exist
    for ddl in [
        f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.lp_module_runs` (run_id STRING, slug STRING, page_url STRING, consent_id STRING, created_ts TIMESTAMP, status STRING, elapsed_ms INT64, error_code STRING) PARTITION BY DATE(created_ts)",
        f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.module_results` (run_id STRING, slug STRING, summary_text STRING, result_json JSON, expires_at TIMESTAMP) PARTITION BY DATE(expires_at)",
    ]:
        bq.query(ddl).result()
    # fetch running
    sql = f"""
      SELECT run_id, slug FROM `{PROJECT}.{DATASET}.lp_module_runs`
      WHERE status='running' LIMIT 20
    """
    rows = list(bq.query(sql).result())
    if not rows:
        print('[runner] No running module runs found.')
        return 0
    for r in rows:
        run_id, slug = r['run_id'], r['slug']
        print(f'[runner] Processing {run_id} {slug}')
        start = time.time()
        out = run_connector(slug, {})
        expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        bq.insert_rows_json(f"{PROJECT}.{DATASET}.module_results", [{
            'run_id': run_id,
            'slug': slug,
            'summary_text': out.get('summary_text',''),
            'result_json': json.dumps(out.get('result_json',{})),
            'expires_at': expires,
        }])
        elapsed = int((time.time() - start)*1000)
        bq.insert_rows_json(f"{PROJECT}.{DATASET}.lp_module_runs", [{
            'run_id': run_id,
            'slug': slug,
            'page_url': None,
            'consent_id': None,
            'created_ts': now_iso(),
            'status': 'done',
            'elapsed_ms': elapsed,
            'error_code': None,
        }])
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

