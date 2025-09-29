#!/usr/bin/env python3
"""
Meta → BigQuery loader (schema ensure + ingestion).

Standardized schema: <dataset>.meta_ad_performance
  - date (DATE), campaign_id STRING, adset_id STRING, ad_id STRING,
    impressions INT64, clicks INT64, cost FLOAT64, conversions FLOAT64, revenue FLOAT64,
    ctr FLOAT64, cvr FLOAT64, avg_cpc FLOAT64, name_hash STRING

Auth: Provide either a user access token (OAuth login; 60‑day) or a Business System User token.
Env:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  - META_ACCESS_TOKEN (required)
  - META_ACCOUNT_ID (e.g., 1234567890; without "act_")
  - OPTIONAL: META_API_VERSION (default: v21.0), AELP2_REDACT_TEXT=1 to store only name hashes

Usage:
  python3 -m AELP2.pipelines.meta_to_bq --start 2025-08-01 --end 2025-09-15 --account 1234567890
"""

import os
import argparse
import hashlib
import json
from datetime import datetime, timedelta
import time
import random
from typing import Dict, List, Any

import requests
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.meta_ad_performance"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("adset_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
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


def ensure_table_by_place(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.meta_ad_performance_by_place"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("adset_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("publisher_platform", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("placement", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("device", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("impressions", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("clicks", "INT64", mode="NULLABLE"),
            bigquery.SchemaField("cost", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("revenue", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("cvr", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("avg_cpc", "FLOAT64", mode="NULLABLE"),
        ]
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="date"
        )
        bq.create_table(table)
        print(f"Created {table_id}")
        return table_id


def delete_range(bq: bigquery.Client, table_id: str, start: str, end: str):
    q = f"""
    DELETE FROM `{table_id}`
    WHERE date BETWEEN DATE(@start) AND DATE(@end)
    """
    job = bq.query(q, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start", "STRING", start),
            bigquery.ScalarQueryParameter("end", "STRING", end),
        ]
    ))
    job.result()


def _pick_conversion(actions: List[Dict[str, Any]]) -> float:
    """Select the primary conversion count for Meta.

    Default behavior: prioritize PURCHASE events over LEAD or higher-funnel actions
    to ensure CAC reflects actual sales. You can override via env META_PRIMARY_ACTION:
      - 'purchase' (default)
      - 'lead'
      - 'initiate_checkout'
    """
    if not actions:
        return 0.0
    primary = os.getenv("META_PRIMARY_ACTION", "purchase").strip().lower()
    by_type = {a.get("action_type"): float(a.get("value", 0.0)) for a in actions}

    prefs_map = {
        "purchase": [
            "offsite_conversion.fb_pixel_purchase",
            "purchase",
            "omni_purchase",
            "onsite_web_purchase",
            "onsite_web_app_purchase",
            "web_in_store_purchase",
        ],
        "lead": [
            "lead",
            "offsite_conversion.fb_pixel_lead",
        ],
        "initiate_checkout": [
            "offsite_conversion.fb_pixel_initiate_checkout",
            "initiate_checkout",
            "omni_initiated_checkout",
            "onsite_web_initiate_checkout",
        ],
    }
    preferred = prefs_map.get(primary, prefs_map["purchase"])  # fallback to purchase
    for t in preferred:
        if t in by_type:
            return by_type[t]
    # Final fallback: 0 (do not sum mixed actions to avoid inflating conversions)
    return 0.0


def _pick_revenue(action_values: List[Dict[str, Any]]) -> float:
    """Select revenue tied to purchases only.

    We restrict to purchase-valued actions to avoid mixing with custom values.
    """
    if not action_values:
        return 0.0
    preferred = [
        "offsite_conversion.fb_pixel_purchase",
        "purchase",
        "omni_purchase",
        "onsite_web_purchase",
        "onsite_web_app_purchase",
        "web_in_store_purchase",
    ]
    by_type = {a.get("action_type"): float(a.get("value", 0.0)) for a in action_values}
    for t in preferred:
        if t in by_type:
            return by_type[t]
    # Fallback: 0 (do not sum heterogenous values)
    return 0.0


def fetch_insights(account_id: str, token: str, start: str, end: str, api_version: str,
                   by_place: bool=False) -> List[Dict[str, Any]]:
    # Ensure account id format
    act_id = account_id
    if not str(act_id).startswith("act_"):
        act_id = f"act_{account_id}"

    url = f"https://graph.facebook.com/{api_version}/{act_id}/insights"
    fields = [
        "date_start", "date_stop", "ad_id", "adset_id", "campaign_id",
        "impressions", "clicks", "spend", "ctr", "cpc", "ad_name",
        "actions", "action_values"
    ]
    params = {
        "access_token": token,
        "level": "ad",
        "time_increment": 1,
        "fields": ",".join(fields),
        "time_range": json.dumps({"since": start, "until": end}),
        "limit": 500,
    }
    if by_place:
        params["breakdowns"] = "publisher_platform,impression_device,platform_position"

    out: List[Dict[str, Any]] = []
    backoff = 1.5
    attempts = 0
    max_attempts = 8
    next_url_cache = None
    while True:
        try:
            resp = requests.get(url, params=params, timeout=60)
        except Exception as e:
            if attempts < max_attempts:
                sleep = backoff ** attempts + random.uniform(0, 0.5)
                time.sleep(sleep)
                attempts += 1
                continue
            raise

        if resp.status_code != 200:
            try:
                j = resp.json()
            except Exception:
                j = {}
            err = (j.get('error') or {})
            code = err.get('code')
            subcode = err.get('error_subcode')
            msg = err.get('message') or resp.text
            # Application-level rate limits or throttling: code 4, 17, 613
            if resp.status_code in (429, 400, 403) and code in (4, 17, 613):
                if attempts < max_attempts:
                    sleep = backoff ** attempts + random.uniform(0, 0.5)
                    time.sleep(sleep)
                    attempts += 1
                    continue
            raise RuntimeError(f"Meta Insights error {resp.status_code} code={code} subcode={subcode}: {msg}")
        # success path resets attempts
        attempts = 0
        data = resp.json()
        for row in data.get("data", []):
            date = row.get("date_start")
            impressions = int(float(row.get("impressions", 0) or 0))
            clicks = int(float(row.get("clicks", 0) or 0))
            cost = float(row.get("spend", 0) or 0)
            ctr = float(row.get("ctr", 0) or 0)
            cpc = float(row.get("cpc", 0) or 0)
            actions = row.get("actions") or []
            action_values = row.get("action_values") or []
            conversions = _pick_conversion(actions)
            revenue = _pick_revenue(action_values)
            cvr = (conversions / clicks) if clicks else 0.0
            ad_name = row.get("ad_name") or ""
            redact_all = os.getenv("AELP2_REDACT_TEXT", "1") == "1"
            name_hash = hashlib.sha256(ad_name.encode()).hexdigest() if redact_all else None

            base = {
                "date": date,
                "campaign_id": str(row.get("campaign_id")),
                "adset_id": str(row.get("adset_id")) if row.get("adset_id") else None,
                "ad_id": str(row.get("ad_id")),
                "impressions": impressions,
                "clicks": clicks,
                "cost": cost,
                "conversions": conversions,
                "revenue": revenue,
                "ctr": ctr if ctr else (clicks / impressions * 100.0 if impressions else 0.0),
                "cvr": cvr,
                "avg_cpc": cpc if cpc else (cost / clicks if clicks else 0.0),
            }
            if by_place:
                base.update({
                    "publisher_platform": row.get("publisher_platform"),
                    "placement": row.get("platform_position"),
                    "device": row.get("impression_device"),
                })
            else:
                base.update({"name_hash": name_hash})
            out.append(base)

        paging = data.get("paging", {})
        next_url = paging.get("next")
        if not next_url:
            break
        # For subsequent pages, use full next URL and clear params
        url, params = next_url, {}

    return out


def date_slices(start: str, end: str, days_per_slice: int = 7) -> List[tuple[str, str]]:
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = s
    while cur <= e:
        slice_end = min(cur + timedelta(days=days_per_slice - 1), e)
        out.append((cur.isoformat(), slice_end.isoformat()))
        cur = slice_end + timedelta(days=1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", help="YYYY-MM-DD; default 7 days ago")
    p.add_argument("--end", help="YYYY-MM-DD; default yesterday")
    p.add_argument("--account", help="Meta ad account id (no act_). Defaults to META_ACCOUNT_ID")
    p.add_argument("--by_placement", action="store_true", help="Ingest breakdown by publisher_platform/placement/device")
    args = p.parse_args()

    today = datetime.utcnow().date()
    start = args.start or (today - timedelta(days=7)).isoformat()
    end = args.end or (today - timedelta(days=1)).isoformat()
    # Validate dates
    datetime.strptime(start, "%Y-%m-%d")
    datetime.strptime(end, "%Y-%m-%d")

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    token = os.getenv('META_ACCESS_TOKEN')
    if not token:
        raise RuntimeError('Set META_ACCESS_TOKEN (user or system user token with ads_read).')
    account = args.account or os.getenv('META_ACCOUNT_ID')
    if not account:
        raise RuntimeError('Provide --account or set META_ACCOUNT_ID')
    api_version = os.getenv('META_API_VERSION', 'v21.0')

    bq = bigquery.Client(project=project)
    if args.by_placement:
        table_id = ensure_table_by_place(bq, project, dataset)
    else:
        table_id = ensure_table(bq, project, dataset)

    total = 0
    for s1, e1 in date_slices(start, end, days_per_slice=7 if args.by_placement else 14):
        rows = fetch_insights(account, token, s1, e1, api_version, by_place=args.by_placement)
        if not rows:
            print(f"No rows for {s1}..{e1}")
            continue
        delete_range(bq, table_id, s1, e1)
        # Insert in chunks to avoid payload limits
        for i in range(0, len(rows), 5000):
            chunk = rows[i:i+5000]
            errors = bq.insert_rows_json(table_id, chunk)
            if errors:
                raise RuntimeError(f"BQ insert errors for slice {s1}..{e1}: {errors}")
            total += len(chunk)
        print(f"Inserted {len(rows)} rows into {table_id} for {s1}..{e1}")
    print(f"Done. Inserted total {total} rows into {table_id} for {start}..{end}")


if __name__ == '__main__':
    main()
