#!/usr/bin/env python3
"""
Build GA4 → Pacer metrics daily table in BigQuery using the native GA4 export dataset (events_*).

Inputs:
  - Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  - Env: GA4_EXPORT_DATASET (e.g., ga360-bigquery-datashare.analytics_308028264). If not set, --export-dataset must be provided.
  - Mapping file: AELP2/config/ga4_pacer_mapping.yaml (same shape as ga4_pacer_to_bq.py)

Outputs:
  - <project>.<dataset>.ga4_pacer_daily (DATE + INT columns per mapped metric)

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training
  export GA4_EXPORT_DATASET=ga360-bigquery-datashare.analytics_308028264
  python3 AELP2/scripts/ga4_export_pacer_to_bq.py --days 400 [--mapping AELP2/config/ga4_pacer_mapping.yaml]
"""

import os
import argparse
from datetime import date, timedelta
from typing import Dict, Any, List

import yaml
from google.cloud import bigquery


def build_clause_for_spec(spec: Dict[str, Any]) -> str:
    ev = spec.get("event_name")
    if not ev:
        raise ValueError("missing event_name in mapping spec")
    parts: List[str] = [f"event_name = '{ev}'"]
    device_category = spec.get("device_category")
    if device_category:
        parts.append(f"device.category = '{device_category}'")
    contains_page = spec.get("contains_page") or []
    if contains_page:
        import re
        pat = "|".join([re.escape(s) for s in contains_page if s])
        if pat:
            parts.append(
                "EXISTS (SELECT 1 FROM UNNEST(event_params) ep "
                "WHERE ep.key = 'page_location' AND REGEXP_CONTAINS(ep.value.string_value, r'(" + pat + ")'))"
            )
    return " AND ".join(parts)


def build_where_clause(metric: str, spec: Dict[str, Any]) -> str:
    # Support either a single spec or an OR-list of specs
    if isinstance(spec.get("or"), list) and spec["or"]:
        clauses = [f"({build_clause_for_spec(s)})" for s in spec["or"]]
        return "(" + " OR ".join(clauses) + ")"
    return build_clause_for_spec(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="AELP2/config/ga4_pacer_mapping.yaml")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--export-dataset", default=os.getenv("GA4_EXPORT_DATASET", ""))
    args = ap.parse_args()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not (project and dataset):
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET")
    export_ds = args.export_dataset
    if not export_ds:
        raise SystemExit("Provide --export-dataset or set GA4_EXPORT_DATASET")

    with open(args.mapping, "r") as f:
        mapping = yaml.safe_load(f)

    client = bigquery.Client(project=project)

    # Build union of dates across all metric queries
    all_dates: set[str] = set()
    metric_to_series: Dict[str, Dict[str, int]] = {}

    start = (date.today() - timedelta(days=args.days)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")

    for metric, spec in mapping.items():
        try:
            where = build_where_clause(metric, spec)
        except ValueError:
            continue
        distinct_user = bool(spec.get("distinct_user", False))
        agg = "COUNT(DISTINCT user_pseudo_id)" if distinct_user else "COUNT(1)"
        q = f"""
            SELECT PARSE_DATE('%Y%m%d', event_date) AS date, {agg} AS cnt
            FROM `{export_ds}.events_*`
            WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
              AND {where}
            GROUP BY date
            ORDER BY date
        """
        job = client.query(q, location="US")
        series: Dict[str, int] = {}
        for r in job.result():
            d = r["date"].isoformat()
            c = int(r["cnt"] or 0)
            series[d] = c
            all_dates.add(d)
        metric_to_series[metric] = series

    # Prepare rows for ga4_pacer_daily
    rows: List[Dict[str, Any]] = []
    for d in sorted(all_dates):
        row: Dict[str, Any] = {"date": d}
        for m in mapping.keys():
            row[m] = int(metric_to_series.get(m, {}).get(d, 0))
        rows.append(row)

    # Write to training dataset (us-central1)
    table_id = f"{project}.{dataset}.ga4_pacer_daily"
    # schema: date + INT columns for mapping keys
    schema = [bigquery.SchemaField("date", "DATE", mode="REQUIRED")] + [
        bigquery.SchemaField(k, "INT64", mode="NULLABLE") for k in mapping.keys()
    ]
    try:
        client.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field="date")
    client.create_table(table)
    job = client.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"))
    job.result()
    print(f"Wrote {len(rows)} rows to {table_id} from export {export_ds}")


if __name__ == "__main__":
    main()
