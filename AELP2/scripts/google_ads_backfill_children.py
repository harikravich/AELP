#!/usr/bin/env python3
"""
Enumerate child Google Ads customer accounts under the MCC and backfill ads_campaign_performance
with capacity metrics (impression share and lost-IS/top-IS) for each child.

Env required:
  - GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
    GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC), GOOGLE_ADS_CUSTOMER_ID (MCC)
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET

Usage:
  python3 -m AELP2.scripts.google_ads_backfill_children --start 2025-06-01 --end 2025-09-17 [--limit 10]
"""
import argparse
import os
from typing import List

from AELP2.pipelines.google_ads_to_bq import run as ingest_run

try:
    from google.ads.googleads.client import GoogleAdsClient
    ADS_OK = True
except Exception as e:
    GoogleAdsClient = None  # type: ignore
    ADS_OK = False


def list_child_customer_ids() -> List[str]:
    if not ADS_OK:
        raise RuntimeError('google-ads not installed')
    cfg = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "use_proto_plus": True,
    }
    login_cid = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login_cid:
        cfg["login_customer_id"] = login_cid.replace("-", "")
    client = GoogleAdsClient.load_from_dict(cfg)
    ga_service = client.get_service("GoogleAdsService")
    query = (
        "SELECT customer_client.client_customer, customer_client.id, customer_client.level, "
        "customer_client.manager, customer_client.descriptive_name, customer_client.status "
        "FROM customer_client WHERE customer_client.status = 'ENABLED'"
    )
    request = client.get_type("SearchGoogleAdsRequest")
    request.customer_id = os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "").replace("-", "")
    request.query = query
    children: List[str] = []
    for row in ga_service.search(request=request):
        cc = row.customer_client
        if bool(cc.manager):
            continue
        cid = str(cc.id)
        if cid and cid not in children:
            children.append(cid)
    return children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    kids = list_child_customer_ids()
    if args.limit and len(kids) > args.limit:
        kids = kids[:args.limit]
    print(f"[ads] Found {len(kids)} child accounts: {kids}")

    ok = 0
    for cid in kids:
        try:
            print(f"[ads] Ingesting child {cid} {args.start}..{args.end}")
            ingest_run(args.start, args.end, customer_id=cid)
            ok += 1
        except Exception as e:
            print(f"[ads] Child {cid} failed: {e}")
            continue
    print(f"[ads] Completed {ok}/{len(kids)} children")


if __name__ == '__main__':
    main()

