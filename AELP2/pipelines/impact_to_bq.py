#!/usr/bin/env python3
"""
Impact.com (Impact Radius) → BigQuery loader.

Pulls Advertiser Reports via Impact.com REST API using Basic Auth
(Account SID + Auth Token) and writes normalized rows to
<project>.<dataset>.impact_partner_performance.

Auth (env):
  IMPACT_ACCOUNT_SID, IMPACT_AUTH_TOKEN

BQ (env):
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Usage examples:
  # List available advertiser reports (IDs & names)
  python3 -m AELP2.pipelines.impact_to_bq.py --list

  # Inspect columns for a report id
  python3 -m AELP2.pipelines.impact_to_bq.py --report adv_partner_performance --describe

  # Ingest partner performance for a date window
  python3 -m AELP2.pipelines.impact_to_bq.py \
    --report adv_partner_performance --start 2025-08-01 --end 2025-09-15

Notes:
  - Report IDs differ by account; pick a Partner-by-Day or Performance-by-Partner report.
  - We auto-map common fields (Date, Partner/MediaPartner, Impressions, Clicks, Actions,
    Payout, Revenue). Unknown columns are ignored.
"""

import os
import sys
import time
import json
import csv
import io
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests
from google.cloud import bigquery


API_BASE = "https://api.impact.com"


def _auth() -> Tuple[str, str, str]:
    """Return (mode, sid, token).

    Modes:
      - 'basic': use IMPACT_ACCOUNT_SID + IMPACT_AUTH_TOKEN (HTTP Basic)
      - 'bearer': use IMPACT_BEARER_TOKEN (Authorization: Bearer ...)
    We still require SID to build the Advertisers/{sid} path even for bearer.
    """
    sid = os.getenv("IMPACT_ACCOUNT_SID")
    bearer = os.getenv("IMPACT_BEARER_TOKEN")
    basic_token = os.getenv("IMPACT_AUTH_TOKEN")
    # Prefer Basic if both present — more reliable across Advertiser endpoints
    if sid and basic_token:
        return "basic", sid, basic_token
    if bearer:
        if not sid:
            raise SystemExit("Set IMPACT_ACCOUNT_SID along with IMPACT_BEARER_TOKEN.")
        return "bearer", sid, bearer
    raise SystemExit("Set either (IMPACT_ACCOUNT_SID + IMPACT_AUTH_TOKEN) or (IMPACT_ACCOUNT_SID + IMPACT_BEARER_TOKEN).")


def list_reports() -> List[Dict[str, Any]]:
    mode, sid, token = _auth()
    url = f"{API_BASE}/Advertisers/{sid}/Reports"
    out: List[Dict[str, Any]] = []
    page = 1
    while True:
        headers = {"Accept": "application/json"}
        params = {"Page": page, "PageSize": 20000}
        if mode == "basic":
            resp = requests.get(url, auth=(sid, token), headers=headers, params=params, timeout=60)
        else:
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}", **headers}, params=params, timeout=60)
        if resp.status_code != 200:
            raise SystemExit(f"Reports list error {resp.status_code}: {resp.text}")
        data = resp.json()
        # Accept both modern and legacy shapes
        if isinstance(data, dict):
            if isinstance(data.get("Reports"), list):
                out.extend(data["Reports"])
            elif isinstance(data.get("_embedded", {}).get("reports"), list):
                out.extend(data["_embedded"]["reports"])
        # Pagination: prefer explicit @nextpageuri; fallback to _links.next
        next_uri = data.get("@nextpageuri") if isinstance(data, dict) else None
        if not next_uri and isinstance(data, dict):
            nxt = data.get("_links", {}).get("next")
            if isinstance(nxt, dict):
                next_uri = nxt.get("href")
        if not next_uri:
            break
        page += 1
    return out


def report_metadata(report_id: str) -> Dict[str, Any]:
    mode, sid, token = _auth()
    url = f"{API_BASE}/Advertisers/{sid}/Reports/{report_id}/MetaData"
    headers = {"Accept": "application/json"}
    if mode == "basic":
        resp = requests.get(url, auth=(sid, token), headers=headers, timeout=60)
    else:
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}", **headers}, timeout=60)
    if resp.status_code != 200:
        raise SystemExit(f"Report metadata error {resp.status_code}: {resp.text}")
    return resp.json()


def _parse_extra_params() -> Dict[str, Any]:
    extras_env = os.getenv('IMPACT_EXTRA_PARAMS', '').strip()
    out: Dict[str, Any] = {}
    if not extras_env:
        return out
    # Accept comma- or ampersand-separated key=value pairs
    pairs = []
    for chunk in extras_env.replace('&', ',').split(','):
        if '=' in chunk:
            k, v = chunk.split('=', 1)
            k = k.strip()
            v = v.strip()
            if k:
                out[k] = v
    return out


def fetch_report_csv(report_id: str, start: str, end: str) -> str:
    mode, sid, token = _auth()
    # Many advertiser reports accept DateStart/DateEnd; format YYYY-MM-DD
    # Use CSV for simpler column handling
    url = f"{API_BASE}/Advertisers/{sid}/Reports/{report_id}.csv"
    # Impact often expects MM/DD/YYYY; support multiple param key styles
    def _mdy(s: str) -> str:
        y, m, d = s.split('-')
        return f"{m}/{d}/{y}"
    candidates = [
        ({"DateStart": _mdy(start), "DateEnd": _mdy(end)}, "mdy DateStart/DateEnd"),
        ({"Start Date": _mdy(start), "End Date": _mdy(end)}, "mdy Start Date/End Date"),
        ({"START_DATE": start, "END_DATE": end}, "iso START_DATE/END_DATE"),
        ({"DateStart": start, "DateEnd": end}, "iso DateStart/DateEnd"),
    ]
    # Optionally enrich with common filters (Event Type/Sale, Program/SubID) via env IMPACT_EXTRA_PARAMS
    extra = _parse_extra_params()
    subaid = os.getenv("IMPACT_SUBAID")
    headers = {"Accept": "text/csv"}
    for base_params, label in candidates:
        params = dict(base_params)
        params["PageSize"] = 20000
        if subaid:
            params["SUBAID"] = subaid
        if extra:
            params.update(extra)
        if mode == "basic":
            resp = requests.get(url, auth=(sid, token), params=params, headers=headers, timeout=120)
        else:
            resp = requests.get(url, headers={"Authorization": f"Bearer {token}", **headers}, params=params, timeout=120)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
    raise SystemExit(f"Report fetch error: no non-empty CSV body returned for any date param style. Last status {resp.status_code}.")


def normalize_rows(csv_text: str) -> List[Dict[str, Any]]:
    # Try to map common columns; tolerate different casings
    reader = csv.DictReader(io.StringIO(csv_text))
    out: List[Dict[str, Any]] = []
    for r in reader:
        cols = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}

        def pick(*names, default=None):
            for n in names:
                v = cols.get(n)
                if v is not None and v != "":
                    return v
            return default

        date = pick("date", "day", "date_start", "date_")
        partner_id = pick("partner id", "partner_id", "media partner id", "media_partner_id")
        partner = pick("partner", "partner name", "media partner", "media_partner")
        impressions = pick("impressions", default="0")
        clicks = pick("clicks", default="0")
        actions = pick("actions", "conversions", default="0")
        payout = pick("payout", "publisher payout", "commission", default="0")
        revenue = pick("revenue", "brand revenue", "sale amount", default="0")

        # Normalize types
        def to_int(x):
            try:
                return int(float(x))
            except Exception:
                return 0

        def to_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        if not date:
            # skip malformed row
            continue

        out.append({
            "date": date,
            "partner_id": partner_id or None,
            "partner": partner or None,
            "impressions": to_int(impressions),
            "clicks": to_int(clicks),
            "actions": to_float(actions),
            "payout": to_float(payout),
            "revenue": to_float(revenue),
        })
    return out


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.impact_partner_performance"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("partner_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("partner", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("impressions", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("clicks", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("actions", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("payout", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("revenue", "FLOAT64", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date"
        )
        bq.create_table(table)
        print(f"Created {table_id}")
        return table_id


def delete_range(bq: bigquery.Client, table_id: str, start: str, end: str):
    q = f"DELETE FROM `{table_id}` WHERE date BETWEEN DATE(@s) AND DATE(@e)"
    job = bq.query(
        q,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("s", "STRING", start),
                bigquery.ScalarQueryParameter("e", "STRING", end),
            ]
        ),
    )
    job.result()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="List available reports and exit")
    p.add_argument("--report", help="Report ID (e.g., adv_partner_performance)")
    p.add_argument("--describe", action="store_true", help="Show report metadata and exit")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    args = p.parse_args()

    if args.list:
        reps = list_reports()
        if not reps:
            print("No reports found or insufficient permissions.")
            return
        for r in reps:
            rid = r.get("id") or r.get("reportId") or r.get("name")
            nm = r.get("name") or r.get("title")
            print(f"{rid}\t{nm}")
        return

    if not args.report:
        raise SystemExit("--report is required (use --list to discover IDs)")

    if args.describe:
        meta = report_metadata(args.report)
        print(json.dumps(meta, indent=2))
        return

    if not args.start or not args.end:
        raise SystemExit("Provide --start and --end (YYYY-MM-DD)")
    # Validate dates
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET")

    csv_text = fetch_report_csv(args.report, args.start, args.end)
    rows = normalize_rows(csv_text)
    if not rows:
        print("No rows returned for given report/date window.")
        return

    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    delete_range(bq, table_id, args.start, args.end)
    errors = bq.insert_rows_json(table_id, rows)
    if errors:
        raise SystemExit(f"BQ insert errors: {errors}")
    print(f"Inserted {len(rows)} rows into {table_id} for {args.start}..{args.end}")


if __name__ == "__main__":
    main()
