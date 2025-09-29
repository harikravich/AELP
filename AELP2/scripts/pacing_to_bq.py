#!/usr/bin/env python3
"""
Load an internal pacing CSV into BigQuery for KPI tie-out.

Expected minimal columns (case-insensitive):
  - date: YYYY-MM-DD (or MM/DD/YYYY)
  - cost or spend: numeric (can include commas)

Optional columns (detected automatically if present):
  - platform (e.g., google_ads, meta, bing, impact)
  - channel/source/medium, campaign, adset, ad
  - clicks, impressions, conversions, revenue

Writes to: <project>.<dataset>.pacing_daily
  date DATE, platform STRING, cost FLOAT64, clicks INT64, impressions INT64,
  conversions FLOAT64, revenue FLOAT64

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training
  python3 AELP2/scripts/pacing_to_bq.py --file AELP2/data/pacing.csv

Notes:
  - CSV only (if you have Excel, export to CSV first).
  - The loader replaces rows for the date range present in the file (idempotent).
"""

import os
import csv
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

from google.cloud import bigquery


def to_int(x: str) -> int:
    try:
        return int(float((x or '0').replace(',', '')))
    except Exception:
        return 0


def to_float(x: str) -> float:
    try:
        return float((x or '0').replace(',', ''))
    except Exception:
        return 0.0


def parse_csv(path: str) -> Tuple[List[Dict[str, Any]], str, str]:
    rows_out: List[Dict[str, Any]] = []
    min_date, max_date = None, None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for r in reader:
            cols = { (k or '').strip().lower(): (v or '').strip() for k, v in r.items() }

            def pick(*names, default=None):
                for n in names:
                    v = cols.get(n)
                    if v not in (None, ''):
                        return v
                return default

            date_raw = pick('date', 'day', 'reporting date')
            if not date_raw:
                continue
            date_norm = None
            for fmt in ('%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y', '%m/%d/%Y %H:%M', '%d/%m/%Y', '%d/%m/%Y %H:%M'):
                try:
                    date_norm = datetime.strptime(date_raw, fmt).strftime('%Y-%m-%d')
                    break
                except Exception:
                    pass
            if not date_norm:
                continue

            platform = pick('platform', 'channel', 'source', 'network')
            # Accept common spend column variants from internal pacing sheets
            cost = to_float(pick('cost', 'spend', 'amount spent', 'amount', 'direct spend', 'media spend', 'ad spend', 'marketing spend', 'total spend'))
            clicks = to_int(pick('clicks', 'clicks (all)'))
            impressions = to_int(pick('impressions'))
            conversions = to_float(pick('conversions', 'purchases', 'leads', 'actions'))
            revenue = to_float(pick('revenue', 'purchase conversion value', 'conversion value', 'value'))

            rows_out.append({
                'date': date_norm,
                'platform': platform or None,
                'cost': cost if cost != 0 else None,
                'clicks': clicks or None,
                'impressions': impressions or None,
                'conversions': conversions or None,
                'revenue': revenue or None,
            })

            if not min_date or date_norm < min_date:
                min_date = date_norm
            if not max_date or date_norm > max_date:
                max_date = date_norm

    if not rows_out:
        raise SystemExit('No usable rows found; ensure CSV has at least date and cost/spend columns.')
    return rows_out, min_date, max_date


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.pacing_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('platform', 'STRING', mode='NULLABLE'),
            bigquery.SchemaField('cost', 'FLOAT64', mode='NULLABLE'),
            bigquery.SchemaField('clicks', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('impressions', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('conversions', 'FLOAT64', mode='NULLABLE'),
            bigquery.SchemaField('revenue', 'FLOAT64', mode='NULLABLE'),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(table)
        print(f'Created {table_id}')
        return table_id


def delete_range(bq: bigquery.Client, table_id: str, start: str, end: str) -> None:
    q = f"DELETE FROM `{table_id}` WHERE date BETWEEN DATE(@s) AND DATE(@e)"
    job = bq.query(q, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter('s', 'STRING', start),
        bigquery.ScalarQueryParameter('e', 'STRING', end),
    ]))
    job.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to pacing CSV (export from Excel if needed)')
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not (project and dataset):
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    rows, start, end = parse_csv(args.file)
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    delete_range(bq, table_id, start, end)
    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        raise SystemExit(f'BQ insert errors: {errors}')
    print(f'Inserted {len(rows)} pacing rows into {table_id} for {start}..{end}')


if __name__ == '__main__':
    main()
