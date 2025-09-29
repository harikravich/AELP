#!/usr/bin/env python3
"""
Reproducibility snapshot: fetch last training_run config and print commands/env to rehydrate.
"""
import os
from google.cloud import bigquery


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print('Missing env; skipping')
        return
    bq = bigquery.Client(project=project)
    sql = f"""
      SELECT * FROM `{project}.{dataset}.training_runs`
      ORDER BY start_time DESC
      LIMIT 1
    """
    rows = list(bq.query(sql).result())
    if not rows:
        print('No training_runs found')
        return
    r = dict(rows[0])
    print('Rehydrate with env and flags:', r.get('configuration') or '{}')


if __name__ == '__main__':
    main()

