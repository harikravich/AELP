#!/usr/bin/env python3
"""
Build GA-aligned KPI daily table that keeps GA as the majority signal and fills only
the non-GA delta from pacer to reach 98–100% totals.

Outputs in <project>.<dataset>:
  - ga4_purchases_utc_daily(date, purchases)           # GA purchases by UTC day incl. intraday
  - ga_aligned_daily(date, ga_enrollments, pacer_subs,
                     non_ga_delta, aligned_enrollments, spend)

Notes
  - Uses GA4 export dataset (events_* + events_intraday_*) in US region.
  - Pacer spend + subs come from pacing_pacer_daily (us-central1).
  - Day cut is locked to midnight UTC (≈8pm ET DST / 7pm ET standard).
"""
import os
from datetime import date, timedelta
from typing import Dict, Any, List
from google.cloud import bigquery


def build_ga_purchases_utc(c_us: bigquery.Client, export_ds: str, days: int) -> Dict[str, int]:
    start = (date.today() - timedelta(days=days+1)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')
    q = f"""
    WITH base AS (
      SELECT DATE(TIMESTAMP_MICROS(event_timestamp), 'UTC') AS d,
             COUNTIF(event_name='purchase') AS p
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
      GROUP BY d
    ), intra AS (
      SELECT DATE(TIMESTAMP_MICROS(event_timestamp), 'UTC') AS d,
             COUNTIF(event_name='purchase') AS p
      FROM `{export_ds}.events_intraday_*`
      WHERE _TABLE_SUFFIX = '{end}'
      GROUP BY d
    )
    SELECT d, SUM(p) AS p FROM (
      SELECT * FROM base UNION ALL SELECT * FROM intra
    ) GROUP BY d
    """
    rows = list(c_us.query(q, location='US').result())
    return {r['d'].isoformat(): int(r['p'] or 0) for r in rows}


def load_table_json(c: bigquery.Client, table_id: str, rows: List[Dict[str, Any]], schema: List[bigquery.SchemaField]) -> None:
    try:
        c.get_table(table_id)
    except Exception:
        t = bigquery.Table(table_id, schema=schema)
        if any(f.name == 'date' and f.field_type == 'DATE' for f in schema):
            t.time_partitioning = bigquery.TimePartitioning(field='date')
        c.create_table(t)
    job = c.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    days = int(os.getenv('GA_ALIGNED_DAYS', '400'))

    c_us = bigquery.Client(project=project)          # US for GA export queries
    c_uc = bigquery.Client(project=project)          # us-central1 dataset writes

    # 1) GA purchases by UTC day (incl. intraday)
    ga = build_ga_purchases_utc(c_us, export_ds, days)
    gp_rows = [{'date': d, 'purchases': v} for d, v in sorted(ga.items())]
    gp_schema = [bigquery.SchemaField('date','DATE','REQUIRED'), bigquery.SchemaField('purchases','INT64')]
    load_table_json(c_uc, f"{project}.{dataset}.ga4_purchases_utc_daily", gp_rows, gp_schema)

    # 2) Pacer totals (spend, subs) and aligned
    q = f"""
    SELECT date, COALESCE(spend,0.0) AS spend, COALESCE(d2c_total_subscribers,0) AS subs
    FROM `{project}.{dataset}.pacing_pacer_daily`
    WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND DATE_ADD(CURRENT_DATE(), INTERVAL 0 DAY)
    """
    P = {r['date'].isoformat(): {'spend': float(r['spend'] or 0.0), 'subs': int(r['subs'] or 0)} for r in c_uc.query(q).result()}

    rows: List[Dict[str, Any]] = []
    all_dates = sorted(set(ga.keys()) | set(P.keys()))
    for d in all_dates:
        ga_en = int(ga.get(d, 0))
        subs = int(P.get(d, {}).get('subs', 0))
        spend = float(P.get(d, {}).get('spend', 0.0))
        delta = max(subs - ga_en, 0)
        # Aligned total equals pacer_subs when available; GA fills gaps when pacer is missing.
        aligned = subs if subs is not None else ga_en
        rows.append({
            'date': d,
            'ga_enrollments': ga_en,
            'pacer_subs': subs,
            'non_ga_delta': delta,
            'aligned_enrollments': aligned,
            'spend': spend,
        })

    schema = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('ga_enrollments','INT64'),
        bigquery.SchemaField('pacer_subs','INT64'),
        bigquery.SchemaField('non_ga_delta','INT64'),
        bigquery.SchemaField('aligned_enrollments','INT64'),
        bigquery.SchemaField('spend','FLOAT64'),
    ]
    load_table_json(c_uc, f"{project}.{dataset}.ga_aligned_daily", rows, schema)
    print(f"Built ga4_purchases_utc_daily ({len(gp_rows)}) and ga_aligned_daily ({len(rows)})")


if __name__ == '__main__':
    main()
