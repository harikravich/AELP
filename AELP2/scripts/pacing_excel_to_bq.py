#!/usr/bin/env python3
"""
Load a multi-sheet pacing Excel workbook to BigQuery (month tabs supported).

Maps common columns:
  - Date → date (YYYY-MM-DD)
  - Direct Spend/Spend/Cost/Amount → cost (FLOAT)
  - D2C Total Subscribers (or Post Trial Subscribers) → conversions (FLOAT)

Writes to: <project>.<dataset>.pacing_daily

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training
  python3 AELP2/scripts/pacing_excel_to_bq.py --file "AELP2/data/Pacing 2025-1.xlsx"
"""

import os
import re
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from google.cloud import bigquery


def clean_money(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    s = s.replace(',', '')
    s = re.sub(r'[^0-9.\-]', '', s)
    try:
        return float(s) if s not in ('', '-', '.') else 0.0
    except Exception:
        return 0.0


def parse_date(x: Any) -> str | None:
    if pd.isna(x):
        return None
    try:
        # pandas handles Excel serials and strings
        d = pd.to_datetime(x, errors='coerce')
        if pd.isna(d):
            return None
        dt = d.date()
        # Ignore bogus epoch-era dates
        if dt < datetime(2000,1,1).date():
            return None
        return dt.isoformat()
    except Exception:
        return None


def extract_rows_from_sheet(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols = {str(c).strip(): c for c in df.columns}
    # Build case-insensitive lookup
    def find_col(*names: str) -> str | None:
        lower = {str(k).strip().lower(): v for k, v in cols.items()}
        for n in names:
            v = lower.get(n.lower())
            if v is not None:
                return v
        return None

    date_col = find_col('Date')
    spend_col = find_col('Direct Spend', 'Spend', 'Cost', 'Amount spent', 'Amount')
    conv_col = find_col('D2C Total Subscribers', 'Post Trial Subscribers')
    ft_col = find_col('Free Trial Starts')
    app_ft_col = find_col('App Trial Starts', 'Mobile Trial Starts')
    d2p_col = find_col('D2P Starts', 'D2P\nStarts')
    post_trial_col = find_col('Post Trial Subscribers')
    mobile_sub_col = find_col('Mobile Subscribers', 'Mobile\nSubscribers')

    if date_col is None or spend_col is None:
        return []

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        d = parse_date(r.get(date_col))
        if not d:
            continue
        cost = clean_money(r.get(spend_col))
        if cost == 0.0 and (r.get(spend_col) in (None, '', 0)):
            # Require some cost indicator; skip non-data rows
            continue
        conv = None
        if conv_col is not None:
            try:
                conv = float(str(r.get(conv_col)).replace(',', '')) if r.get(conv_col) not in (None, '') else None
            except Exception:
                conv = None
        # Optional pacer metrics
        def to_int(x):
            try:
                return int(float(str(x).replace(',', '')))
            except Exception:
                return None
        free_trial = to_int(r.get(ft_col)) if ft_col else None
        app_trial = to_int(r.get(app_ft_col)) if app_ft_col else None
        d2p = to_int(r.get(d2p_col)) if d2p_col else None
        post_trial = to_int(r.get(post_trial_col)) if post_trial_col else None
        mobile_sub = to_int(r.get(mobile_sub_col)) if mobile_sub_col else None
        out.append({
            'date': d,
            'platform': None,
            'cost': cost if cost != 0 else None,
            'clicks': None,
            'impressions': None,
            'conversions': conv,
            'revenue': None,
            'free_trial_starts': free_trial,
            'app_trial_starts': app_trial,
            'd2p_starts': d2p,
            'post_trial_subscribers': post_trial,
            'mobile_subscribers': mobile_sub,
        })
    return out


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


def ensure_pacer_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.pacing_pacer_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('spend', 'FLOAT64', mode='NULLABLE'),
            bigquery.SchemaField('free_trial_starts', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('app_trial_starts', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('d2p_starts', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('post_trial_subscribers', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('mobile_subscribers', 'INT64', mode='NULLABLE'),
            bigquery.SchemaField('d2c_total_subscribers', 'INT64', mode='NULLABLE'),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(table)
        print(f'Created {table_id}')
        return table_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to pacing Excel file (.xlsx)')
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not (project and dataset):
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    xlsx = args.file
    if not os.path.isfile(xlsx):
        raise SystemExit(f'File not found: {xlsx}')

    # Read all sheets
    book = pd.read_excel(xlsx, sheet_name=None)
    all_rows: List[Dict[str, Any]] = []
    for name, df in book.items():
        rows = extract_rows_from_sheet(df)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise SystemExit('No usable rows found across sheets. Check column names and formats.')

    # Determine date range and load
    dates = sorted({r['date'] for r in all_rows if r.get('date')})
    start, end = dates[0], dates[-1]

    bq = bigquery.Client(project=project)
    # Rebuild the table from this workbook (simplest and avoids streaming buffer issues)
    table_id = f"{project}.{dataset}.pacing_daily"
    try:
        bq.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    table_id = ensure_table(bq, project, dataset)
    # Base table: only the core columns
    base_rows = [
        {k: v for k, v in r.items() if k in ('date','platform','cost','clicks','impressions','conversions','revenue')}
        for r in all_rows
    ]
    job = bq.load_table_from_json(base_rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()
    # Rich pacer metrics table
    pacer_id = ensure_pacer_table(bq, project, dataset)
    pacer_rows = []
    for r in all_rows:
        pacer_rows.append({
            'date': r['date'],
            'spend': r.get('cost'),
            'free_trial_starts': r.get('free_trial_starts'),
            'app_trial_starts': r.get('app_trial_starts'),
            'd2p_starts': r.get('d2p_starts'),
            'post_trial_subscribers': r.get('post_trial_subscribers'),
            'mobile_subscribers': r.get('mobile_subscribers'),
            'd2c_total_subscribers': int(r['conversions']) if r.get('conversions') is not None else None,
        })
    job2 = bq.load_table_from_json(pacer_rows, pacer_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job2.result()
    print(f'Loaded {len(base_rows)} base rows into {table_id} and {len(pacer_rows)} rows into {pacer_id} for {start}..{end} from {len(book)} sheets')


if __name__ == '__main__':
    main()
