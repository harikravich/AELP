#!/usr/bin/env python3
"""
End-to-end CLI for safely setting up Google Ads API access for a personal Gmail Ads account
without affecting existing Aura credentials.

Features:
- Uses OAuth2 Device Code flow (no browser on VM required)
- Reuses existing OAuth client (Client ID/Secret) from another project
- Mints a new refresh token for the Gmail user (does not change other users)
- Saves credentials to a dedicated env file (keeps Aura creds untouched)
- Tests API connectivity
- Optional: creates a paused Search campaign with RSA, keywords, and UTM suffix

Usage examples:

  # 1) Interactive flow to mint token and save env
  python3 scripts/google_ads_gmail_setup.py \
    --developer-token YOUR_DEV_TOKEN \
    --client-id YOUR_OAUTH_CLIENT_ID \
    --client-secret YOUR_OAUTH_CLIENT_SECRET \
    --customer-id 9704174968 \
    --env-out AELP2/config/.google_ads_credentials.gmail.env

  # 2) Also create a paused campaign (safe)
  python3 scripts/google_ads_gmail_setup.py \
    --developer-token YOUR_DEV_TOKEN \
    --client-id YOUR_OAUTH_CLIENT_ID \
    --client-secret YOUR_OAUTH_CLIENT_SECRET \
    --customer-id 9704174968 \
    --create-campaign \
    --campaign-name "Aura Sandbox - Identity Protection (Gmail)" \
    --daily-budget 50 \
    --final-url "https://your-sandbox-landing.example.com/?stream=gmail" \
    --utm-suffix "utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword}&stream=gmail"

Notes:
- Keep GOOGLE_ADS_LOGIN_CUSTOMER_ID unset for a standalone Gmail account.
- If OAuth consent screen is in Testing, add the Gmail as a Test User; otherwise publish the app.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, List

try:
    import requests
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"], stdout=subprocess.DEVNULL)
    import requests


DEVICE_CODE_ENDPOINT = "https://oauth2.googleapis.com/device/code"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
ADS_SCOPE = "https://www.googleapis.com/auth/adwords"


@dataclass
class OAuthClient:
    client_id: str
    client_secret: str
    scope: str = ADS_SCOPE


def get_device_code(client: OAuthClient) -> dict:
    resp = requests.post(DEVICE_CODE_ENDPOINT, data={
        "client_id": client.client_id,
        "scope": client.scope,
    })
    resp.raise_for_status()
    return resp.json()


def poll_for_token(client: OAuthClient, device_code: str, interval: int, timeout_s: int) -> dict:
    start = time.time()
    while True:
        if time.time() - start > timeout_s:
            raise TimeoutError("Device code authorization timed out")

        r = requests.post(TOKEN_ENDPOINT, data={
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        })
        try:
            js = r.json()
        except Exception:
            r.raise_for_status()
            js = {}

        if "error" not in js:
            return js

        err = js.get("error")
        if err in ("authorization_pending", "slow_down"):
            time.sleep(interval)
            continue
        raise RuntimeError(f"OAuth error: {err} {js.get('error_description','')}")


def ensure_google_ads_lib():
    try:
        import google.ads.googleads  # noqa: F401
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-ads"], stdout=subprocess.DEVNULL)


def google_ads_client(config: dict):
    ensure_google_ads_lib()
    from google.ads.googleads.client import GoogleAdsClient
    return GoogleAdsClient.load_from_dict(config)


def test_connect(config: dict, customer_id: str) -> None:
    client = google_ads_client(config)
    ga_service = client.get_service("GoogleAdsService")
    query = "SELECT customer.id, customer.descriptive_name FROM customer LIMIT 1"
    for batch in ga_service.search_stream(customer_id=customer_id, query=query):
        for row in batch.results:
            print(f"OK: {row.customer.id} {row.customer.descriptive_name}")
            return
    print("OK: Query succeeded (no rows)")


def create_paused_campaign(
    config: dict,
    customer_id: str,
    campaign_name: str,
    daily_budget: int,
    final_url: str,
    utm_suffix: str,
    headlines: Optional[List[str]] = None,
    descriptions: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> None:
    import uuid
    from google.ads.googleads.errors import GoogleAdsException

    client = google_ads_client(config)
    budget_svc = client.get_service("CampaignBudgetService")
    camp_svc = client.get_service("CampaignService")
    ag_svc = client.get_service("AdGroupService")
    aga_svc = client.get_service("AdGroupAdService")
    agc_svc = client.get_service("AdGroupCriterionService")

    headlines = (headlines or [])[:15]
    descriptions = (descriptions or [])[:4]
    keywords = keywords or []

    # Budget
    bud_op = client.get_type("CampaignBudgetOperation")
    bud = bud_op.create
    bud.name = f"Gmail Budget {uuid.uuid4()}"
    bud.amount_micros = int(daily_budget) * 1_000_000
    # Some versions require setting delivery method; safe to try
    try:
        bud.delivery_method = client.enums.BudgetDeliveryMethodEnum.STANDARD
    except Exception:
        pass
    bud_res = budget_svc.mutate_campaign_budgets(customer_id=customer_id, operations=[bud_op])
    budget_rn = bud_res.results[0].resource_name

    # Campaign (paused)
    camp_op = client.get_type("CampaignOperation")
    camp = camp_op.create
    camp.name = campaign_name
    camp.advertising_channel_type = client.enums.AdvertisingChannelTypeEnum.SEARCH
    camp.status = client.enums.CampaignStatusEnum.PAUSED
    camp.campaign_budget = budget_rn
    camp.network_settings.target_google_search = True
    camp.network_settings.target_search_network = True
    camp.network_settings.target_content_network = False
    camp.network_settings.target_partner_search_network = False
    # Manual CPC (try; API versions differ)
    try:
        camp.manual_cpc.CopyFrom(client.get_type("ManualCpc"))
    except Exception:
        pass
    camp.final_url_suffix = utm_suffix
    camp_res = camp_svc.mutate_campaigns(customer_id=customer_id, operations=[camp_op])
    camp_rn = camp_res.results[0].resource_name

    # Ad group
    ag_op = client.get_type("AdGroupOperation")
    ag = ag_op.create
    ag.name = f"{campaign_name} - Ad Group 1"
    ag.campaign = camp_rn
    ag.status = client.enums.AdGroupStatusEnum.ENABLED
    ag.cpc_bid_micros = 1_500_000  # $1.50
    ag_res = ag_svc.mutate_ad_groups(customer_id=customer_id, operations=[ag_op])
    ag_rn = ag_res.results[0].resource_name

    # RSA (paused)
    aga_op = client.get_type("AdGroupAdOperation")
    aga = aga_op.create
    aga.ad_group = ag_rn
    aga.status = client.enums.AdGroupAdStatusEnum.PAUSED
    ad = aga.ad
    ad.final_urls.append(final_url)
    rsa = ad.responsive_search_ad
    for h in headlines:
        a = client.get_type("AdTextAsset"); a.text = h; rsa.headlines.append(a)
    for d in descriptions:
        a = client.get_type("AdTextAsset"); a.text = d; rsa.descriptions.append(a)
    aga_svc.mutate_ad_group_ads(customer_id=customer_id, operations=[aga_op])

    # Keywords
    kw_ops = []
    for kw in keywords:
        op = client.get_type("AdGroupCriterionOperation")
        c = op.create
        c.ad_group = ag_rn
        c.status = client.enums.AdGroupCriterionStatusEnum.ENABLED
        c.keyword.text = kw
        c.keyword.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
        kw_ops.append(op)
    if kw_ops:
        agc_svc.mutate_ad_group_criteria(customer_id=customer_id, operations=kw_ops)

    print("Created paused Search campaign with RSA + keywords.")


def write_env_file(path: str, developer_token: str, client_id: str, client_secret: str, refresh_token: str, customer_id: str, login_customer_id: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("GOOGLE_ADS_DEVELOPER_TOKEN=" + developer_token + "\n")
        f.write("GOOGLE_ADS_CLIENT_ID=" + client_id + "\n")
        f.write("GOOGLE_ADS_CLIENT_SECRET=" + client_secret + "\n")
        f.write("GOOGLE_ADS_REFRESH_TOKEN=" + refresh_token + "\n")
        if login_customer_id:
            f.write("GOOGLE_ADS_LOGIN_CUSTOMER_ID=" + login_customer_id + "\n")
        f.write("GOOGLE_ADS_CUSTOMER_ID=" + customer_id + "\n")
    print(f"Saved credentials to {path}")


def build_ads_config(developer_token: str, client_id: str, client_secret: str, refresh_token: str, login_customer_id: Optional[str] = None) -> dict:
    cfg = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "use_proto_plus": True,
    }
    if login_customer_id:
        cfg["login_customer_id"] = login_customer_id
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Setup Google Ads for Gmail account without breaking Aura creds")
    parser.add_argument("--developer-token", required=True, help="Google Ads developer token")
    parser.add_argument("--client-id", required=True, help="OAuth client ID (reuse existing)")
    parser.add_argument("--client-secret", required=True, help="OAuth client secret")
    parser.add_argument("--customer-id", required=True, help="Gmail Ads customer ID (10 digits, no dashes)")
    parser.add_argument("--login-customer-id", help="Optional MCC login customer ID (leave unset for standalone Gmail)")
    parser.add_argument("--env-out", default=None, help="Where to save env file (default: AELP2/config/.google_ads_credentials.gmail.env if exists, else ./.google_ads_credentials.gmail.env)")
    parser.add_argument("--refresh-token", default=None, help="Optional: provide an existing refresh token to skip device-code auth")
    parser.add_argument("--no-test", action="store_true", help="Skip API connectivity test")
    parser.add_argument("--create-campaign", action="store_true", help="Create a paused Search campaign with RSA and keywords")
    parser.add_argument("--campaign-name", default="Aura Sandbox - Identity Protection (Gmail)")
    parser.add_argument("--daily-budget", type=int, default=50, help="Daily budget in USD")
    parser.add_argument("--final-url", default="https://your-sandbox-landing.example.com/?stream=gmail")
    parser.add_argument("--utm-suffix", default="utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword}&stream=gmail")
    parser.add_argument("--headline", action="append", default=[
        "Identity Guard® Official Site",
        "#1 Identity Theft Protection",
        "Compare Pricing Plans",
        "Premium Identity Protection",
    ], help="Add a headline (can repeat)")
    parser.add_argument("--description", action="append", default=[
        "Protect your identity with 24/7 monitoring and $1M coverage.",
        "Save up to 33% today. Fast alerts. US-based support.",
    ], help="Add a description (can repeat)")
    parser.add_argument("--keyword", action="append", default=[
        "identity theft protection",
        "identity protection",
        "identity theft monitoring",
    ], help="Add a keyword (can repeat)")

    args = parser.parse_args()

    # Determine env output path
    if args.env_out:
        env_path = args.env_out
    else:
        env_path = "AELP2/config/.google_ads_credentials.gmail.env" if os.path.isdir("AELP2/config") else ".google_ads_credentials.gmail.env"

    # 1) Get or mint refresh token for Gmail user
    if args.refresh_token:
        refresh_token = args.refresh_token.strip()
        if not refresh_token:
            raise ValueError("--refresh-token provided but empty")
        print("Using provided refresh token (skipping device-code auth).")
    else:
        client = OAuthClient(client_id=args.client_id, client_secret=args.client_secret)
        dc = get_device_code(client)
        print("\nOn your laptop, visit:", dc["verification_url"])
        print("Enter this code:", dc["user_code"])  # Keep this visible for the user
        print("\nWaiting for approval... (Ctrl+C to abort)")

        js = poll_for_token(client, dc["device_code"], interval=int(dc.get("interval", 5)), timeout_s=int(dc.get("expires_in", 600)))
        refresh_token = js.get("refresh_token")
        if not refresh_token:
            raise RuntimeError(f"No refresh_token in response: {js}")
        print("\nReceived refresh token for Gmail user.")

    # 2) Write env file (keeps separate from Aura)
    write_env_file(
        path=env_path,
        developer_token=args.developer_token,
        client_id=args.client_id,
        client_secret=args.client_secret,
        refresh_token=refresh_token,
        customer_id=args.customer_id,
        login_customer_id=args.login_customer_id,
    )

    # Build in-memory config for immediate use
    cfg = build_ads_config(
        developer_token=args.developer_token,
        client_id=args.client_id,
        client_secret=args.client_secret,
        refresh_token=refresh_token,
        login_customer_id=args.login_customer_id,
    )

    # 3) Test connectivity (optional)
    if not args.no_test:
        try:
            test_connect(cfg, args.customer_id)
        except Exception as e:
            print("Connectivity test failed:", e)
            print("- Ensure Gmail is added as Test User (if consent is Testing)")
            print("- Ensure T&Cs accepted in Google Ads UI for this account")
            print("- If using MCC, set --login-customer-id to the MCC ID")
            sys.exit(2)

    # 4) Optional: create paused campaign
    if args.create_campaign:
        try:
            create_paused_campaign(
                cfg,
                customer_id=args.customer_id,
                campaign_name=args.campaign_name,
                daily_budget=args.daily_budget,
                final_url=args.final_url,
                utm_suffix=args.utm_suffix,
                headlines=args.headline,
                descriptions=args.description,
                keywords=args.keyword,
            )
        except Exception as e:
            print("Create campaign failed:", e)
            print("- Confirm billing is set up and account T&Cs accepted")
            print("- Check developer token access level")
            sys.exit(3)

    print("\nAll done. To use these creds in a session:")
    print(f"  set -a; source {env_path}; set +a")
    print("  unset GOOGLE_ADS_LOGIN_CUSTOMER_ID  # keep unset for standalone Gmail")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
