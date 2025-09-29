#!/usr/bin/env python3
"""Seed bandit_posteriors from recent ads performance (heuristic).

Computes CAC mean and an approximate 95% CI per cell_key (campaign:ad_id) using
last 14 days of ads_ad_performance and writes to bandit_posteriors.
"""
import os, math, time
from datetime import date, timedelta
from google.cloud import bigquery  # type: ignore
from datetime import datetime, timezone

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
DATASET = os.environ.get('BIGQUERY_TRAINING_DATASET')

assert PROJECT and DATASET, 'Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET'

client = bigquery.Client(project=PROJECT)

START = (date.today() - timedelta(days=14)).isoformat()
END = date.today().isoformat()

sql = f'''
SELECT campaign_id, ad_id,
       SUM(cost_micros)/1e6 AS cost,
       SUM(conversions) AS conv
FROM `{PROJECT}.{DATASET}.ads_ad_performance`
WHERE DATE(date) BETWEEN @start AND @end
GROUP BY campaign_id, ad_id
HAVING conv > 0 AND cost > 0
ORDER BY cost DESC
LIMIT 200
'''

rows = client.query(sql, job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter('start', 'DATE', START),
        bigquery.ScalarQueryParameter('end', 'DATE', END),
    ]
)).result()

to_insert = []
ts = bigquery.ScalarQueryParameter

for r in rows:
    cac = float(r.cost) / float(r.conv)
    # Wilson-style approximation for CAC uncertainty
    # Assume each conversion has variance; use sqrt(n) scaling as placeholder
    n = max(1.0, float(r.conv))
    se = max(1.0, cac) / math.sqrt(n)  # very rough
    ci_low = max(0.0, cac - 1.96 * se)
    ci_high = cac + 1.96 * se
    cell_key = f"campaign:{r.campaign_id}:ad:{r.ad_id}"
    to_insert.append({
        'ts': bigquery.ScalarQueryParameter(None, 'TIMESTAMP', None),  # placeholder
        'cell_key': cell_key,
        'metric': 'cac',
        'mean': float(cac),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'samples': int(n),
    })

# Insert with current timestamp
table = client.dataset(DATASET).table('bandit_posteriors')
now_iso = datetime.now(timezone.utc).isoformat()
errors = client.insert_rows_json(table, [
    {
        'ts': now_iso,
        'cell_key': r['cell_key'],
        'metric': r['metric'],
        'mean': r['mean'],
        'ci_low': r['ci_low'],
        'ci_high': r['ci_high'],
        'samples': r['samples'],
    } for r in to_insert
])
if errors:
    raise SystemExit(f'Insert errors: {errors}')

print(f'Inserted {len(to_insert)} bandit_posteriors rows.')
