#!/usr/bin/env python3
"""
Load a Meta Ads Manager CSV export into BigQuery (fallback when API access is blocked).

Input: CSV exported at ad level with daily breakdown from Ads Manager.
Maps common column names to the standard table schema used by meta_to_bq.py:
  - date (DATE), campaign_id, adset_id, ad_id, impressions, clicks, cost,
    conversions, revenue, ctr, cvr, avg_cpc, name_hash

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training
  python3 AELP2/scripts/meta_csv_to_bq.py --file /path/to/export.csv

Notes:
  - The script is tolerant to column label differences (“Clicks (all)”, “Amount spent (USD)”, etc.).
  - Ad names are hashed if AELP2_REDACT_TEXT=1 (default).
  - It deletes and replaces rows for the covered date range in the target table.
"""

import os
import csv
import io
import sys
import argparse
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple

from google.cloud import bigquery


def detect_delimiter(sample: bytes) -> str:
    # Try to guess delimiter between comma and tab
    text = sample.decode(errors="ignore")
    if "\t" in text and text.count("\t") > text.count(","):
        return "\t"
    return ","


def parse_csv(path: str) -> Tuple[List[Dict[str, Any]], str, str]:
    # Read a small sample to detect delimiter and encoding
    with open(path, "rb") as f:
        sample = f.read(4096)
    delimiter = detect_delimiter(sample)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)

    # Normalize rows
    out: List[Dict[str, Any]] = []
    min_date, max_date = None, None
    redact_all = os.getenv("AELP2_REDACT_TEXT", "1") == "1"

    def to_int(x):
        try:
            return int(float((x or "0").replace(",", "")))
        except Exception:
            return 0

    def to_float(x):
        try:
            return float((x or "0").replace(",", ""))
        except Exception:
            return 0.0

    # Map likely column names → our schema
    for r in rows:
        cols = { (k or "").strip().lower(): (v or "").strip() for k, v in r.items() }

        def pick(*names, default=None):
            for n in names:
                v = cols.get(n)
                if v not in (None, ""):
                    return v
            return default

        date = pick("date", "reporting starts", "date start", "date_start")
        if not date:
            # Skip row if no date
            continue
        # Normalize date to YYYY-MM-DD
        date_norm = None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                date_norm = datetime.strptime(date, fmt).strftime("%Y-%m-%d")
                break
            except Exception:
                pass
        if not date_norm:
            # Unknown format; skip
            continue

        ad_id = pick("ad id", "ad_id", "ad id ", "ad")
        adset_id = pick("ad set id", "adset id", "ad_set_id")
        campaign_id = pick("campaign id", "campaign_id")

        impressions = to_int(pick("impressions"))
        clicks = to_int(pick("clicks (all)", "link clicks", "clicks", "clicks all"))
        spend = to_float(pick("amount spent (usd)", "amount spent", "spend", "cost"))
        ctr = to_float(pick("ctr (all)", "ctr"))
        cpc = to_float(pick("cpc (all)", "cpc", "cost per click (all)"))

        # Conversions and revenue vary; prefer purchases
        conversions = to_float(pick(
            "purchases", "results", "conversions", "leads", "lead"))
        revenue = to_float(pick(
            "purchases conversion value", "purchase conversion value",
            "conversion value", "value"))

        ad_name = pick("ad name", "ad") or ""
        name_hash = hashlib.sha256(ad_name.encode()).hexdigest() if redact_all else None

        row = {
            "date": date_norm,
            "campaign_id": campaign_id or None,
            "adset_id": adset_id or None,
            "ad_id": ad_id or None,
            "impressions": impressions or None,
            "clicks": clicks or None,
            "cost": spend or None,
            "conversions": conversions or None,
            "revenue": revenue or None,
            "ctr": ctr or ( (clicks / impressions * 100.0) if impressions else None),
            "cvr": (conversions / clicks) if (clicks and conversions is not None) else None,
            "avg_cpc": cpc or ( (spend / clicks) if clicks else None),
            "name_hash": name_hash,
        }
        out.append(row)

        if not min_date or date_norm < min_date:
            min_date = date_norm
        if not max_date or date_norm > max_date:
            max_date = date_norm

    if not out:
        raise SystemExit("No usable rows found in CSV. Ensure ad-level, daily breakdown export.")
    return out, min_date, max_date


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.meta_ad_performance"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("campaign_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("adset_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ad_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("impressions", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("clicks", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("cost", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("revenue", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("cvr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("avg_cpc", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("name_hash", "STRING", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date"
        )
        bq.create_table(table)
        print(f"Created {table_id}")
        return table_id


def delete_range(bq: bigquery.Client, table_id: str, start: str, end: str) -> None:
    q = f"DELETE FROM `{table_id}` WHERE date BETWEEN DATE(@s) AND DATE(@e)"
    job = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("s", "STRING", start),
            bigquery.ScalarQueryParameter("e", "STRING", end),
        ]
    ))
    job.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to Ads Manager CSV export")
    args = ap.parse_args()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET")

    rows, start, end = parse_csv(args.file)
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    delete_range(bq, table_id, start, end)
    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        raise SystemExit(f"BQ insert errors: {errors}")
    print(f"Inserted {len(rows)} rows into {table_id} for {start}..{end}")


if __name__ == "__main__":
    main()

