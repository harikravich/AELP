#!/usr/bin/env python3
"""Bandits Posteriors Writer (heuristic P0)
Reads explore_cells and writes a simple CAC posterior summary to bandit_posteriors.
Replace with MABWiser TS later.
"""
import os, json, random, time
from datetime import datetime, timezone
from google.cloud import bigquery  # type: ignore

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
DATASET = os.environ.get('BIGQUERY_TRAINING_DATASET')

def now():
    return datetime.now(timezone.utc).isoformat()

def main():
    if not PROJECT or not DATASET:
        print('Missing env'); return 2
    bq = bigquery.Client(project=PROJECT)
    try:
        rows = list(bq.query(f"SELECT cell_key, angle, audience, channel, lp, offer, spend, conversions FROM `{PROJECT}.{DATASET}.explore_cells` WHERE last_seen >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)").result())
    except Exception:
        rows = []
    if not rows:
        print('No explore cells.'); return 0
    # Create table
    bq.query(f"CREATE TABLE IF NOT EXISTS `{PROJECT}.{DATASET}.bandit_posteriors` (ts TIMESTAMP, cell_key STRING, metric STRING, mean FLOAT64, ci_low FLOAT64, ci_high FLOAT64, samples INT64) PARTITION BY DATE(ts)").result()
    data = []
    for r in rows:
        spend = float(r.get('spend') or 0)
        conv = float(r.get('conversions') or 0)
        samples = max(1, int(conv))
        cac = (spend / conv) if conv > 0 else 999.0
        ci = max(5.0, cac*0.2)
        data.append({ 'ts': now(), 'cell_key': r['cell_key'], 'metric': 'cac', 'mean': cac, 'ci_low': max(0.0, cac-ci), 'ci_high': cac+ci, 'samples': samples })
    bq.insert_rows_json(f"{PROJECT}.{DATASET}.bandit_posteriors", data)
    print(f"Wrote {len(data)} posterior rows")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
