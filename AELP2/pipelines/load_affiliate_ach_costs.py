#!/usr/bin/env python3
"""
Load external affiliate (ACH) payouts into BigQuery to complete partner costs.

Input CSV (optional): AELP2/data/affiliate_ach_costs.csv
Columns (header required): date, partner_id, partner, amount, memo
  - date: YYYY-MM-DD (posting or performance date you want costs recognized)
  - partner_id: Impact MediaId as string (preferred). If empty, we try to map by partner name.
  - partner: Partner name (for mapping / display)
  - amount: positive float USD
  - memo: optional notes

BQ table: <project>.<dataset>.affiliates_external_costs
Schema: date DATE, partner_id STRING, partner STRING, amount FLOAT64, memo STRING, source STRING

Usage:
  python3 AELP2/pipelines/load_affiliate_ach_costs.py

Notes:
  - If the CSV is absent, the loader creates the table (if missing) and exits.
  - You can re-run; rows are upserted by (date, partner_id, partner, amount, memo).
"""
from __future__ import annotations
import os, csv, sys
from typing import List, Dict
from google.cloud import bigquery

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'affiliate_ach_costs.csv')
CSV_PATH = os.path.normpath(CSV_PATH)


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.affiliates_external_costs"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('date','DATE','REQUIRED'),
            bigquery.SchemaField('partner_id','STRING','NULLABLE'),
            bigquery.SchemaField('partner','STRING','NULLABLE'),
            bigquery.SchemaField('amount','FLOAT','REQUIRED'),
            bigquery.SchemaField('memo','STRING','NULLABLE'),
            bigquery.SchemaField('source','STRING','NULLABLE'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(field='date')
        bq.create_table(t)
        return table_id


def load_csv(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            date = (row.get('date') or '').strip()
            partner_id = (row.get('partner_id') or '').strip() or None
            partner = (row.get('partner') or '').strip() or None
            memo = (row.get('memo') or '').strip() or None
            amt_raw = (row.get('amount') or '').strip()
            try:
                amount = float(amt_raw)
            except Exception:
                continue
            if not date or amount == 0.0:
                continue
            rows.append({'date': date, 'partner_id': partner_id, 'partner': partner, 'amount': amount, 'memo': memo, 'source': 'ach_csv'})
    return rows


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)

    if not os.path.exists(CSV_PATH):
        print(f"[ach] No CSV at {CSV_PATH}; table ensured; nothing to load.")
        return

    rows = load_csv(CSV_PATH)
    if not rows:
        print('[ach] CSV has no valid rows; nothing to load.')
        return

    # Delete overlapping dates to simplify upsert
    dates = sorted({r['date'] for r in rows})
    s, e = dates[0], dates[-1]
    bq.query(f"DELETE FROM `{table_id}` WHERE date BETWEEN DATE('{s}') AND DATE('{e}')").result()
    # Insert in chunks
    chunk = 1000
    for i in range(0, len(rows), chunk):
        batch = rows[i:i+chunk]
        errs = bq.insert_rows_json(table_id, batch)
        if errs:
            raise SystemExit(f"BQ insert errors: {errs[:3]}")
    print(f"[ach] Loaded {len(rows)} rows into {table_id} from {CSV_PATH}")


if __name__ == '__main__':
    main()

