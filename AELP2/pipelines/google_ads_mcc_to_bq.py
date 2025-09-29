#!/usr/bin/env python3
"""
Enumerate all child accounts under an MCC and load Ads performance into BigQuery.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Ads env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List

from AELP2.pipelines.google_ads_to_bq import run as load_campaigns

try:
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
    ADS_AVAILABLE = True
except ImportError as e:
    GoogleAdsClient = None
    GoogleAdsException = Exception
    ADS_AVAILABLE = False
    ADS_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def enumerate_child_accounts() -> List[str]:
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")

    cfg = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "login_customer_id": os.environ["GOOGLE_ADS_LOGIN_CUSTOMER_ID"].replace("-", ""),
        "use_proto_plus": True,
    }
    client = GoogleAdsClient.load_from_dict(cfg)
    ga_service = client.get_service("GoogleAdsService")
    request = client.get_type("SearchGoogleAdsRequest")
    request.customer_id = cfg["login_customer_id"]
    request.query = """
        SELECT
          customer_client.client_customer,
          customer_client.level,
          customer_client.manager,
          customer_client.hidden,
          customer_client.status
        FROM customer_client
        WHERE customer_client.level = 1
          AND customer_client.hidden = FALSE
      """
    resp = ga_service.search(request=request)
    cids: List[str] = []
    for row in resp:
        resource = row.customer_client.client_customer  # e.g., 'customers/1234567890'
        if resource:
            cid = str(resource).split("/")[-1].replace("-", "")
            cids.append(cid)
    return cids


def main(start: str, end: str, list_only: bool = False):
    cids = enumerate_child_accounts()
    if not cids:
        logger.warning("No child accounts found under MCC")
        return
    if list_only:
        print("\n".join(cids))
        return
    logger.info(f"Found {len(cids)} child accounts under MCC; loading {start}..{end}")
    for cid in cids:
        try:
            load_campaigns(start, end, customer_id=cid)
        except Exception as e:
            logger.error(f"Failed to load for customer {cid}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--list-only", action="store_true", help="Only list child customer account IDs")
    args = p.parse_args()
    # Validate dates
    for d in [args.start, args.end]:
        datetime.strptime(d, "%Y-%m-%d")
    main(args.start, args.end, list_only=args.list_only)
