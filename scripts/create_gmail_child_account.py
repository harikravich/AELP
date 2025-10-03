#!/usr/bin/env python3
"""
Create a child (client) Google Ads account under a manager (MCC).

Reads credentials from environment:
- GOOGLE_ADS_DEVELOPER_TOKEN (required)
- OAuth (prefer Gmail): GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN
  Fallback: GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET, GOOGLE_ADS_REFRESH_TOKEN
- GOOGLE_ADS_LOGIN_CUSTOMER_ID (manager CID; defaults to 9704174968 if unset)

Optional env overrides:
- CHILD_NAME (default: Hari Sandbox (Gmail))
- CHILD_TIME_ZONE (default: America/Los_Angeles)
- CHILD_CURRENCY (default: USD)

Prints the new child Customer ID on success.
"""

import os
import sys

def env_or_error(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"Missing required env: {name}", file=sys.stderr)
        sys.exit(2)
    return v

def main():
    # Developer token
    dev_token = env_or_error("GOOGLE_ADS_DEVELOPER_TOKEN")

    # OAuth: prefer Gmail-specific, fallback to generic
    cid = os.getenv("GMAIL_CLIENT_ID") or os.getenv("GOOGLE_ADS_CLIENT_ID")
    csecret = os.getenv("GMAIL_CLIENT_SECRET") or os.getenv("GOOGLE_ADS_CLIENT_SECRET")
    rtoken = os.getenv("GMAIL_REFRESH_TOKEN") or os.getenv("GOOGLE_ADS_REFRESH_TOKEN")
    if not cid or not csecret or not rtoken:
        print("Missing OAuth env. Provide GMAIL_CLIENT_ID/SECRET/REFRESH_TOKEN or GOOGLE_ADS_CLIENT_ID/SECRET/REFRESH_TOKEN", file=sys.stderr)
        sys.exit(2)

    # Manager id
    login_cid = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID", "9704174968").replace("-", "")

    # Lazy import and helpful message if lib missing
    try:
        from google.ads.googleads.client import GoogleAdsClient
    except Exception as e:
        print("google-ads library missing. Install with: python3 -m pip install google-ads", file=sys.stderr)
        print(str(e), file=sys.stderr)
        sys.exit(3)

    cfg = {
        "developer_token": dev_token,
        "client_id": cid,
        "client_secret": csecret,
        "refresh_token": rtoken,
        "login_customer_id": login_cid,
        "use_proto_plus": True,
    }
    client = GoogleAdsClient.load_from_dict(cfg)
    svc = client.get_service("CustomerService")

    # Child account basics
    name = os.getenv("CHILD_NAME", "Hari Sandbox (Gmail)")
    tz = os.getenv("CHILD_TIME_ZONE", "America/Los_Angeles")
    cur = os.getenv("CHILD_CURRENCY", "USD")

    cust = client.get_type("Customer")
    cust.descriptive_name = name
    cust.currency_code = cur
    cust.time_zone = tz
    # Safe tracking template and final suffix (UTMs are added at ad/campaign creation time)
    cust.tracking_url_template = "{lpurl}"
    cust.final_url_suffix = "utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword}&stream=gmail"

    req = client.get_type("CreateCustomerClientRequest")
    req.customer_id = login_cid
    # Assign the constructed Customer message directly
    req.customer_client = cust

    try:
        resp = svc.create_customer_client(request=req)
        child_cid = resp.resource_name.split('/')[-1]
        print("New child Customer ID:", child_cid)
    except Exception as e:
        print(f"CreateCustomerClient failed: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    main()
