#!/usr/bin/env python3
"""
Impact.com entities → BigQuery (MediaPartners, Ads, Deals, Invoices, TrackingValueRequests).

Auth (env): IMPACT_ACCOUNT_SID, IMPACT_AUTH_TOKEN (Basic) or IMPACT_BEARER_TOKEN (Bearer)
BQ (env): GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Usage:
  python3 -m AELP2.pipelines.impact_entities_to_bq.py --entities media_partners,ads,deals,invoices,tvr
"""
from __future__ import annotations
import os, sys, time, json
from typing import Dict, Any, List, Tuple
import requests
from google.cloud import bigquery
from requests.auth import HTTPBasicAuth

API_BASE = "https://api.impact.com"

def _auth() -> Tuple[str, str, str]:
    sid = os.getenv("IMPACT_ACCOUNT_SID")
    bearer = os.getenv("IMPACT_BEARER_TOKEN")
    basic = os.getenv("IMPACT_AUTH_TOKEN")
    # Prefer Basic if both are present — Advertiser endpoints consistently accept Basic
    if sid and basic:
        return ("basic", sid, basic)
    if bearer:
        if not sid:
            raise SystemExit("Set IMPACT_ACCOUNT_SID along with IMPACT_BEARER_TOKEN")
        return ("bearer", sid, bearer)
    raise SystemExit("Set either (IMPACT_ACCOUNT_SID + IMPACT_AUTH_TOKEN) or (IMPACT_ACCOUNT_SID + IMPACT_BEARER_TOKEN)")

def _get(url: str, mode: str, sid: str, token: str, params: Dict[str, Any] | None = None) -> requests.Response:
    headers = {"Accept":"application/json", "Content-Type":"application/json"}
    if mode == "basic":
        return requests.get(url, auth=HTTPBasicAuth(sid, token), headers=headers, params=params or {}, timeout=60)
    return requests.get(url, headers={"Authorization": f"Bearer {token}", **headers}, params=params or {}, timeout=60)

def fetch_paginated(endpoint: str) -> List[Dict[str, Any]]:
    mode, sid, token = _auth()
    base_url = f"{API_BASE}/Advertisers/{sid}/{endpoint}"
    items: List[Dict[str, Any]] = []
    # First request without query params (some endpoints reject explicit Page params)
    r = _get(base_url, mode, sid, token, params=None)
    if r.status_code != 200:
        # Retry with Page/PageSize
        r = _get(base_url, mode, sid, token, params={"Page": 1, "PageSize": 200})
        if r.status_code != 200:
            raise RuntimeError(f"{endpoint} fetch error {r.status_code}: {r.text[:200]}")
    j = r.json()
    def append_from_payload(jobj):
        if isinstance(jobj, dict):
            # Prefer explicit keys; fall back to any list-typed collection present
            for key in ("Partners","MediaPartners","Ads","Deals","Invoices","TrackingValueRequests"):
                if key in jobj and isinstance(jobj[key], list):
                    items.extend(jobj[key])
                    return jobj.get("@nextpageuri")
            # Generic fallback: take the first list value found (excluding link/meta keys)
            for k, v in jobj.items():
                if k.startswith('@') or k.lower() in ("_links","links","paging"): 
                    continue
                if isinstance(v, list):
                    items.extend(v)
                    return jobj.get("@nextpageuri")
        return jobj.get("@nextpageuri") if isinstance(jobj, dict) else None
    next_uri = append_from_payload(j)
    # Follow @nextpageuri if present
    while next_uri:
        next_url = f"{API_BASE}{next_uri}" if next_uri.startswith('/') else next_uri
        r = _get(next_url, mode, sid, token, params=None)
        if r.status_code != 200:
            raise RuntimeError(f"{endpoint} nextpage fetch error {r.status_code}: {r.text[:200]}")
        j = r.json()
        next_uri = append_from_payload(j)
        time.sleep(0.15)
    return items

def load_bq(table: str, rows: List[Dict[str, Any]]):
    if not rows:
        print(f"[impact] {table}: 0 rows (skipped)")
        return
    project = os.getenv("GOOGLE_CLOUD_PROJECT"); dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET")
    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.{table}"
    # Create table if missing with autodetect (JSON load)
    try:
        bq.get_table(table_id)
    except Exception:
        schema = []  # let autodetect
        t = bigquery.Table(table_id, schema=schema)
        bq.create_table(t)
    job = bq.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE', autodetect=True))
    job.result()
    print(f"[impact] Loaded {len(rows)} rows into {table_id}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", default="media_partners,ads,deals,invoices,tvr", help="comma list: media_partners,ads,deals,invoices,tvr")
    args = ap.parse_args()
    ents = [e.strip() for e in args.entities.split(',') if e.strip()]
    mapping = {
        'media_partners': ('MediaPartners', 'impact_media_partners'),
        'ads': ('Ads', 'impact_ads'),
        'deals': ('Deals', 'impact_deals'),
        'invoices': ('Invoices', 'impact_invoices'),
        'tvr': ('TrackingValueRequests', 'impact_tracking_value_requests'),
    }
    for ent in ents:
        if ent not in mapping:
            print(f"Unknown entity: {ent}")
            continue
        endpoint, table = mapping[ent]
        print(f"[impact] Fetching {endpoint} → {table}")
        rows = fetch_paginated(endpoint)
        load_bq(table, rows)

if __name__ == '__main__':
    main()
