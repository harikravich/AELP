#!/usr/bin/env python3
"""
Create a basic Google Ads Search campaign with an Ad Group, keywords, and a Responsive Search Ad (RSA).

Defaults:
- Status PAUSED for safety
- Channel SEARCH, Google Search only (no content network)
- Optional campaign-level Final URL Suffix for UTM parameters

Env required:
- GOOGLE_ADS_DEVELOPER_TOKEN
- GOOGLE_ADS_CLIENT_ID
- GOOGLE_ADS_CLIENT_SECRET
- GOOGLE_ADS_REFRESH_TOKEN
- GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)  [used for manager context]

Args:
  --customer: Target child customer ID (10 digits, no dashes)
  --name: Campaign name
  --daily_budget: Float dollars (e.g., 100.0)
  --final_url: Landing page (e.g., https://example.com)
  --utm_suffix: Optional Final URL Suffix (e.g.,
      utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword})
  --headline: Repeatable; up to 10–15 suggested
  --description: Repeatable; up to 4 suggested
  --keyword: Repeatable; default ['identity theft protection']

Example:
  python -m AELP2.scripts.create_google_ads_search_campaign \
    --customer 7844126439 \
    --name "Aura Sandbox - ID Protection" \
    --daily_budget 100 \
    --final_url https://landing.example.com/offer \
    --utm_suffix "utm_source=google&utm_medium=cpc&utm_campaign={campaignid}&utm_content={creative}&utm_term={keyword}" \
    --headline "Identity Guard® Official Site" \
    --headline "#1 Identity Theft Protection" \
    --headline "Compare Pricing Plans" \
    --description "Protect your identity with 24/7 monitoring and $1M coverage." \
    --description "Save up to 33% today. Fast alerts. US-based support." \
    --keyword "identity theft protection" --keyword "identity protection"
"""

import os
import argparse
from typing import List

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException


def build_client() -> GoogleAdsClient:
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
    return GoogleAdsClient.load_from_dict(cfg)


def create_budget(client: GoogleAdsClient, customer_id: str, daily_budget_dollars: float) -> str:
    budget_svc = client.get_service("CampaignBudgetService")
    op = client.get_type("CampaignBudgetOperation")
    budget = op.create
    budget.name = f"Budget {daily_budget_dollars:.2f} {os.urandom(2).hex()}"
    budget.delivery_method = client.enums.BudgetDeliveryMethodEnum.STANDARD
    budget.amount_micros = int(round(daily_budget_dollars * 1_000_000))
    try:
        budget.explicitly_shared = False
    except Exception:
        pass
    resp = budget_svc.mutate_campaign_budgets(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def create_campaign(
    client: GoogleAdsClient,
    customer_id: str,
    name: str,
    budget_res: str,
    final_url_suffix: str | None,
) -> str:
    campaign_svc = client.get_service("CampaignService")
    op = client.get_type("CampaignOperation")
    c = op.create
    c.name = name
    c.advertising_channel_type = client.enums.AdvertisingChannelTypeEnum.SEARCH
    # Pause on create for safety; enable after inspection
    c.status = client.enums.CampaignStatusEnum.PAUSED
    c.campaign_budget = budget_res
    # Bidding: set Manual CPC explicitly
    try:
        mcpc = client.get_type("ManualCpc")
        mcpc.enhanced_cpc_enabled = False
        c.manual_cpc = mcpc
    except Exception:
        pass
    # Networks: search only
    c.network_settings.target_google_search = True
    c.network_settings.target_search_network = True
    c.network_settings.target_partner_search_network = False
    c.network_settings.target_content_network = False
    if final_url_suffix:
        c.final_url_suffix = final_url_suffix
    # Compliance: explicitly state not EU political advertising
    try:
        c.contains_eu_political_advertising = False
    except Exception:
        pass
    resp = campaign_svc.mutate_campaigns(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def create_ad_group(client: GoogleAdsClient, customer_id: str, campaign_res: str, name: str) -> str:
    ag_svc = client.get_service("AdGroupService")
    op = client.get_type("AdGroupOperation")
    ag = op.create
    ag.name = name
    ag.campaign = campaign_res
    ag.status = client.enums.AdGroupStatusEnum.ENABLED
    # Optional CPC bid; many strategies ignore it. Keep conservative.
    ag.cpc_bid_micros = 200_000  # $0.20
    resp = ag_svc.mutate_ad_groups(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def add_keywords(client: GoogleAdsClient, customer_id: str, ad_group_res: str, keywords: List[str]):
    if not keywords:
        return
    agc_svc = client.get_service("AdGroupCriterionService")
    ops = []
    for kw in keywords:
        op = client.get_type("AdGroupCriterionOperation")
        crit = op.create
        crit.ad_group = ad_group_res
        crit.status = client.enums.AdGroupCriterionStatusEnum.ENABLED
        crit.keyword.text = kw
        crit.keyword.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
        ops.append(op)
    agc_svc.mutate_ad_group_criteria(customer_id=customer_id, operations=ops)


def add_rsa(client: GoogleAdsClient, customer_id: str, ad_group_res: str, final_url: str, headlines: List[str], descriptions: List[str]):
    aga_svc = client.get_service("AdGroupAdService")
    op = client.get_type("AdGroupAdOperation")
    aga = op.create
    aga.status = client.enums.AdGroupAdStatusEnum.PAUSED  # paused for review
    aga.ad_group = ad_group_res
    ad = aga.ad
    ad.final_urls.append(final_url)
    rsa = ad.responsive_search_ad
    for h in headlines[:15]:
        asset = client.get_type("AdTextAsset")
        asset.text = h
        rsa.headlines.append(asset)
    for d in descriptions[:4]:
        asset = client.get_type("AdTextAsset")
        asset.text = d
        rsa.descriptions.append(asset)
    # Domain verification: leave path empty; Google may infer from Final URL
    resp = aga_svc.mutate_ad_group_ads(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customer", required=True, help="Customer ID (10 digits)")
    p.add_argument("--name", required=True, help="Campaign name")
    p.add_argument("--daily_budget", type=float, required=True, help="Daily budget in dollars")
    p.add_argument("--final_url", required=True, help="Landing page final URL")
    p.add_argument("--utm_suffix", default=None, help="Final URL Suffix with UTM params")
    p.add_argument("--headline", action="append", default=[], help="RSA headline (repeatable)")
    p.add_argument("--description", action="append", default=[], help="RSA description (repeatable)")
    p.add_argument("--keyword", action="append", default=[], help="Keyword (repeatable)")
    args = p.parse_args()

    try:
        client = build_client()
        cid = args.customer.replace('-', '')
        budget_res = create_budget(client, cid, args.daily_budget)
        camp_res = create_campaign(client, cid, args.name, budget_res, args.utm_suffix)
        ag_res = create_ad_group(client, cid, camp_res, f"{args.name} - Ad Group 1")
        kws = args.keyword or ["identity theft protection"]
        add_keywords(client, cid, ag_res, kws)
        heads = args.headline or [
            "Identity Guard® Official Site",
            "#1 Identity Theft Protection",
            "Compare Pricing Plans",
            "Premium Identity Protection",
        ]
        descs = args.description or [
            "Protect your identity with 24/7 monitoring and $1M coverage.",
            "Save up to 33% today. Fast alerts. US-based support.",
        ]
        ad_res = add_rsa(client, cid, ag_res, args.final_url, heads, descs)
        print("Created:")
        print("  Budget:", budget_res)
        print("  Campaign:", camp_res)
        print("  Ad Group:", ag_res)
        print("  RSA:", ad_res)
    except GoogleAdsException as e:
        print("Google Ads API error:", e)
        for err in e.failure.errors:
            print(" -", err)
        raise


if __name__ == "__main__":
    main()
