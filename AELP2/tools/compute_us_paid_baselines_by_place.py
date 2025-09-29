#!/usr/bin/env python3
from __future__ import annotations
"""
Compute US Meta paid-event baselines by publisher_platform and placement from
`<project>.<dataset>.meta_ad_performance_by_place`.

Writes: AELP2/reports/us_meta_baselines_by_place.json

Fields per key "{platform}/{placement}": cpm_p10/p50/p90, ctr_p10/p50/p90, cvr_p10/p50/p90,
and totals (imps, clicks, cost, conversions).
"""
import argparse, json, os
from datetime import date, timedelta
from pathlib import Path

from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'us_meta_baselines_by_place.json'


def run(days: int) -> dict:
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET') or os.getenv('BIGQUERY_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET in .env')

    table = f"{project}.{dataset}.meta_ad_performance_by_place"
    bq = bigquery.Client(project=project)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)

    q = f"""
    WITH daily AS (
      SELECT
        DATE(date) AS d,
        COALESCE(publisher_platform, 'unknown') AS platform,
        COALESCE(placement, 'unknown') AS placement,
        SAFE_DIVIDE(SUM(cost), NULLIF(SUM(impressions),0)) * 1000 AS cpm,
        SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
        SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
        SUM(impressions) AS imps,
        SUM(clicks) AS clicks,
        SUM(cost) AS cost,
        SUM(conversions) AS conv
      FROM `{table}`
      WHERE DATE(date) BETWEEN DATE(@start) AND DATE(@end)
      GROUP BY d, platform, placement
    )
    SELECT
      platform,
      placement,
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
    FROM daily
    GROUP BY platform, placement
    """
    job = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('start', 'DATE', start.isoformat()),
            bigquery.ScalarQueryParameter('end', 'DATE', end.isoformat()),
        ]
    ))
    rows = list(job.result())
    out = {
        'start_date': start.isoformat(),
        'end_date': end.isoformat(),
        'items': {}
    }
    for r in rows:
        rec = dict(r)
        key = f"{rec['platform']}/{rec['placement']}"
        # convert decimals
        for k,v in list(rec.items()):
            if hasattr(v, 'as_tuple'):
                rec[k] = float(v)
        out['items'][key] = {
            'cpm_p10': rec['cpm_p10'], 'cpm_p50': rec['cpm_p50'], 'cpm_p90': rec['cpm_p90'],
            'ctr_p10': rec['ctr_p10'], 'ctr_p50': rec['ctr_p50'], 'ctr_p90': rec['ctr_p90'],
            'cvr_p10': rec['cvr_p10'], 'cvr_p50': rec['cvr_p50'], 'cvr_p90': rec['cvr_p90'],
            'imps': rec['imps'], 'clicks': rec['clicks'], 'cost': rec['cost'], 'conversions': rec['conversions'],
            'days': rec['days'],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = run(args.days)
    OUT.write_text(json.dumps(data, indent=2))
    print(json.dumps({'out': str(OUT), 'keys': len(data.get('items', {}))}, indent=2))


if __name__ == '__main__':
    main()

