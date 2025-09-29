#!/usr/bin/env python3
"""
Namespace Refactor Report (stub): scans repo for non-AELP2 imports and prints summary.

Writes `<project>.<dataset>.namespace_refactor_report` with counts per prefix (optional).
"""
import os, re
from datetime import datetime
try:
    from google.cloud import bigquery
    HAVE_BQ = True
except Exception:
    HAVE_BQ = False

PATTERN = re.compile(r'^\s*(?:from|import)\s+([A-Za-z0-9_\.]+)', re.M)


def scan_repo(root: str):
    counts = {}
    for dp, _, files in os.walk(root):
        for f in files:
            if f.endswith('.py') and '/.venv/' not in dp and '/.git/' not in dp:
                p = os.path.join(dp, f)
                try:
                    s = open(p, 'r', encoding='utf-8', errors='ignore').read()
                except Exception:
                    continue
                for m in PATTERN.finditer(s):
                    mod = m.group(1).split('.')[0]
                    if mod and mod not in ('AELP2',):
                        counts[mod] = counts.get(mod, 0) + 1
    return counts


def main():
    root = os.getenv('AELP2_REPO_ROOT', '/workspace')
    counts = scan_repo(root)
    print('Top external prefixes:', sorted(counts.items(), key=lambda x: -x[1])[:20])
    project = os.getenv('GOOGLE_CLOUD_PROJECT'); dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if HAVE_BQ and project and dataset:
        bq = bigquery.Client(project=project)
        table_id = f"{project}.{dataset}.namespace_refactor_report"
        try:
            bq.get_table(table_id)
        except Exception:
            bq.create_table(bigquery.Table(table_id, schema=[
                bigquery.SchemaField('timestamp','TIMESTAMP'),
                bigquery.SchemaField('module','STRING'),
                bigquery.SchemaField('count','INT64'),
            ]))
        rows = [{'timestamp': datetime.utcnow().isoformat(), 'module': m, 'count': c} for m, c in counts.items()]
        if rows:
            bq.insert_rows_json(table_id, rows)
            print(f'wrote {len(rows)} rows to {table_id}')


if __name__ == '__main__':
    main()

