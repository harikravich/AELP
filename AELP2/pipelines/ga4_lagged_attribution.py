#!/usr/bin/env python3
"""
GA4 Lagged Attribution Importer

Computes a lag-aware attribution of GA4 conversions to prior days and writes
results into `<project>.<dataset>.ga4_lagged_attribution`.

The attribution redistributes GA4 daily conversions for each day D backward
to dates D-L where L is within [MIN_LAG_DAYS, MAX_LAG_DAYS] using a
triangular weighting that peaks at PEAK_LAG_DAYS (default 7).

Env vars:
- GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- AELP2_GA4_ATTRIB_MIN_LAG_DAYS (default: 3)
- AELP2_GA4_ATTRIB_MAX_LAG_DAYS (default: 14)
- AELP2_GA4_ATTRIB_PEAK_LAG_DAYS (default: 7)

Input:
- `${dataset}.ga4_daily` view created by create_bq_views.py

Output table schema:
- date (DATE): target date receiving lagged credit
- ga4_conversions_lagged (FLOAT)
- window_start (DATE), window_end (DATE)
- method (STRING)
- computed_at (TIMESTAMP)
- details (JSON)
"""

import os
import argparse
from datetime import datetime, timedelta, date
from typing import Dict, List
import json

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from AELP2.core.ingestion.bq_loader import get_bq_client, insert_rows


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def ensure_table(bq: bigquery.Client, table_id: str):
    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("ga4_conversions_lagged", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("method", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("computed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
    ]
    try:
        bq.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date"
        )
        bq.create_table(table)


def triangular_weights(min_lag: int, max_lag: int, peak_lag: int) -> Dict[int, float]:
    if not (min_lag <= peak_lag <= max_lag):
        peak_lag = (min_lag + max_lag) // 2
    weights: Dict[int, float] = {}
    # Rising slope from min_lag to peak
    for L in range(min_lag, peak_lag + 1):
        weights[L] = (L - min_lag + 1)
    # Falling slope from peak+1 to max_lag
    for L in range(peak_lag + 1, max_lag + 1):
        weights[L] = (max_lag - L + 1)
    total = sum(weights.values()) or 1.0
    for L in list(weights.keys()):
        weights[L] = weights[L] / total
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    dataset = os.environ["BIGQUERY_TRAINING_DATASET"]
    ds = f"{project}.{dataset}"
    bq = get_bq_client()

    min_lag = _env_int("AELP2_GA4_ATTRIB_MIN_LAG_DAYS", 3)
    max_lag = _env_int("AELP2_GA4_ATTRIB_MAX_LAG_DAYS", 14)
    peak_lag = _env_int("AELP2_GA4_ATTRIB_PEAK_LAG_DAYS", 7)
    w = triangular_weights(min_lag, max_lag, peak_lag)

    # Read GA4 daily conversions
    ga4_sql = f"""
        SELECT DATE(date) AS date, SUM(conversions) AS conversions
        FROM `{ds}.ga4_daily`
        WHERE DATE(date) BETWEEN '{args.start}' AND '{args.end}'
        GROUP BY date
        ORDER BY date
    """
    ga4 = {str(r.date): float(r.conversions or 0.0) for r in bq.query(ga4_sql).result()}

    # Build lagged attribution per target date
    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    out: Dict[str, float] = {}
    for d_str, conv in ga4.items():
        obs_date = datetime.strptime(d_str, "%Y-%m-%d").date()
        if conv <= 0:
            continue
        for L, wt in w.items():
            target = obs_date - timedelta(days=L)
            if target < start_d or target > end_d:
                continue
            key = target.isoformat()
            out[key] = out.get(key, 0.0) + conv * wt

    # Write rows
    table_id = f"{ds}.ga4_lagged_attribution"
    ensure_table(bq, table_id)
    computed_at = datetime.utcnow().isoformat()
    window_start = start_d.isoformat()
    window_end = end_d.isoformat()
    rows = [
        {
            "date": d,
            "ga4_conversions_lagged": v,
            "window_start": window_start,
            "window_end": window_end,
            "method": "triangular",
            "computed_at": computed_at,
            "details": json.dumps({
                "min_lag": min_lag,
                "max_lag": max_lag,
                "peak_lag": peak_lag,
            }),
        }
        for d, v in sorted(out.items())
    ]
    if rows:
        insert_rows(bq, table_id, rows)
        print(f"Inserted {len(rows)} rows into {table_id}")
    else:
        print("No lagged attribution rows computed for given window")


if __name__ == "__main__":
    import json  # needed for details JSON
    main()
