#!/usr/bin/env python3
from __future__ import annotations
"""
Compute US Meta baselines from BigQuery and write to reports/us_meta_baselines.json.

Assumes insights were ingested to <project>.<dataset>.meta_ad_performance via
AELP2/pipelines/meta_to_bq.py. If geo fields are absent, we treat the account as US‑focused.

Env required:
  GOOGLE_CLOUD_PROJECT
  BIGQUERY_TRAINING_DATASET

Optional args:
  --days 90   (lookback window)
"""
import argparse, json, os
from datetime import date, timedelta
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'us_meta_baselines.json'


def run(days: int) -> dict:
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET') or os.getenv('BIGQUERY_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET in .env')

    table = f"{project}.{dataset}.meta_ad_performance"
    bq = bigquery.Client(project=project)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)

    q = f"""
    WITH base AS (
      SELECT
        DATE(date) AS d,
        SAFE_DIVIDE(SUM(cost), NULLIF(SUM(impressions),0)) * 1000 AS cpm,
        SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
        SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
        SUM(impressions) AS imps,
        SUM(clicks) AS clicks,
        SUM(cost) AS cost,
        SUM(conversions) AS conv
      FROM `{table}`
      WHERE DATE(date) BETWEEN DATE(@start) AND DATE(@end)
      GROUP BY d
    )
    SELECT
      @start AS start_date,
      @end AS end_date,
      COUNT(*) AS days,
      AVG(cpm) AS cpm_avg,
      APPROX_QUANTILES(cpm, 100)[OFFSET(10)] AS cpm_p10,
      APPROX_QUANTILES(cpm, 100)[OFFSET(50)] AS cpm_p50,
      APPROX_QUANTILES(cpm, 100)[OFFSET(90)] AS cpm_p90,
      AVG(ctr) AS ctr_avg,
      APPROX_QUANTILES(ctr, 100)[OFFSET(10)] AS ctr_p10,
      APPROX_QUANTILES(ctr, 100)[OFFSET(50)] AS ctr_p50,
      APPROX_QUANTILES(ctr, 100)[OFFSET(90)] AS ctr_p90,
      AVG(cvr) AS cvr_avg,
      APPROX_QUANTILES(cvr, 100)[OFFSET(10)] AS cvr_p10,
      APPROX_QUANTILES(cvr, 100)[OFFSET(50)] AS cvr_p50,
      APPROX_QUANTILES(cvr, 100)[OFFSET(90)] AS cvr_p90,
      SUM(imps) AS imps,
      SUM(clicks) AS clicks,
      SUM(cost) AS cost,
      SUM(conv) AS conversions
    FROM base
    """
    job = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('start', 'DATE', start.isoformat()),
            bigquery.ScalarQueryParameter('end', 'DATE', end.isoformat()),
        ]
    ))
    rows = list(job.result())
    if not rows:
        raise SystemExit('No rows returned from meta_ad_performance; check ingestion and dates')
    rec = dict(rows[0])
    # Convert Decimal/Float64 to float
    for k, v in list(rec.items()):
        if hasattr(v, 'as_tuple'):
            rec[k] = float(v)
        if hasattr(v, 'isoformat'):
            rec[k] = v.isoformat()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rec = run(args.days)
    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps({'out': str(OUT), **rec}, indent=2))


if __name__ == '__main__':
    main()
