#!/usr/bin/env python3
"""
Discover accessible Google Ads accounts (including MCC/manager accounts).

Prints customer IDs with basic metadata and whether each is a manager (MCC).

Requirements:
- Env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN

Usage:
  python3 -m AELP2.pipelines.google_ads_discover_accounts
"""

import os
import logging

try:
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
except ImportError as e:
    raise SystemExit(f"google-ads not installed: {e}")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def load_client() -> GoogleAdsClient:
    cfg = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "use_proto_plus": True,
    }
    return GoogleAdsClient.load_from_dict(cfg)


def main() -> int:
    client = load_client()
    customer_service = client.get_service("CustomerService")

    # List accessible customers (returns resource names)
    resp = customer_service.list_accessible_customers()
    resources = list(resp.resource_names)
    if not resources:
        print("No accessible customers found. Check OAuth credentials.")
        return 1

    print("Accessible Google Ads customers:\n")
    # Fetch details for each customer
    details = []
    for rn in resources:
        try:
            cust = customer_service.get_customer(resource_name=rn)
            # Extract ID from resource name 'customers/xxxxxxxxxx'
            cust_id = rn.split("/")[-1]
            details.append({
                "id": cust_id,
                "descriptive_name": getattr(cust, "descriptive_name", ""),
                "currency_code": getattr(cust, "currency_code", ""),
                "time_zone": getattr(cust, "time_zone", ""),
                "manager": bool(getattr(cust, "manager", False)),
            })
        except GoogleAdsException as e:
            logger.warning(f"Failed to describe {rn}: {e}")

    # Managers first
    details.sort(key=lambda d: (not d["manager"], d["id"]))

    for d in details:
        flag = "MCC" if d["manager"] else "ACCT"
        name = d["descriptive_name"] or "(no name)"
        print(f"{flag}\t{d['id']}\t{name}\t{d['currency_code']}\t{d['time_zone']}")

    print("\nTo use MCC, set: export GOOGLE_ADS_LOGIN_CUSTOMER_ID=<MCC_ID>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

