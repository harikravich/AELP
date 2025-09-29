#!/usr/bin/env python3
"""
Add a Search ad group, keywords, and a paused RSA to an existing campaign.

Env required:
- GOOGLE_ADS_DEVELOPER_TOKEN
- GOOGLE_ADS_CLIENT_ID
- GOOGLE_ADS_CLIENT_SECRET
- GOOGLE_ADS_REFRESH_TOKEN
- (optional) GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)

Args:
  --customer: Child customer ID (10 digits)
  --campaign_id: Existing campaign ID
  --adgroup_name: Name for the new ad group
  --final_url: Final URL for the RSA (e.g., https://www.aura.com/identity-theft-protection?stream=gmail)
  --utm_suffix: Optional Final URL suffix to set on the campaign (overwrites existing)
  --headline: Repeatable RSA headline
  --description: Repeatable RSA description
  --keyword: Repeatable phrase‑match keyword (text only; script applies PHRASE type)
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
    lcid = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if lcid:
        cfg["login_customer_id"] = lcid.replace("-", "")
    return GoogleAdsClient.load_from_dict(cfg)


def update_campaign_suffix(client: GoogleAdsClient, customer_id: str, campaign_id: int, utm_suffix: str):
    campaign_service = client.get_service("CampaignService")
    op = client.get_type("CampaignOperation")
    c = op.update
    c.resource_name = campaign_service.campaign_path(customer_id, campaign_id)
    c.final_url_suffix = utm_suffix
    fm = client.get_type("FieldMask")
    fm.paths.append("final_url_suffix")
    op.update_mask = fm
    campaign_service.mutate_campaigns(customer_id=customer_id, operations=[op])


def create_ad_group(client: GoogleAdsClient, customer_id: str, campaign_id: int, name: str) -> str:
    ag_service = client.get_service("AdGroupService")
    op = client.get_type("AdGroupOperation")
    ag = op.create
    ag.name = name
    ag.campaign = ag_service.campaign_path(customer_id, campaign_id)
    ag.status = client.enums.AdGroupStatusEnum.ENABLED
    ag.cpc_bid_micros = 200_000  # $0.20
    resp = ag_service.mutate_ad_groups(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def add_keywords(client: GoogleAdsClient, customer_id: str, ad_group_res: str, keywords: List[str]):
    if not keywords:
        return
    svc = client.get_service("AdGroupCriterionService")
    ops = []
    for kw in keywords:
        op = client.get_type("AdGroupCriterionOperation")
        crit = op.create
        crit.ad_group = ad_group_res
        crit.status = client.enums.AdGroupCriterionStatusEnum.ENABLED
        crit.keyword.text = kw
        crit.keyword.match_type = client.enums.KeywordMatchTypeEnum.PHRASE
        ops.append(op)
    svc.mutate_ad_group_criteria(customer_id=customer_id, operations=ops)


def add_rsa(client: GoogleAdsClient, customer_id: str, ad_group_res: str, final_url: str, headlines: List[str], descriptions: List[str]) -> str:
    svc = client.get_service("AdGroupAdService")
    op = client.get_type("AdGroupAdOperation")
    aga = op.create
    aga.status = client.enums.AdGroupAdStatusEnum.PAUSED
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
    resp = svc.mutate_ad_group_ads(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customer", required=True)
    p.add_argument("--campaign_id", required=True, type=int)
    p.add_argument("--adgroup_name", required=True)
    p.add_argument("--final_url", required=True)
    p.add_argument("--utm_suffix", default=None)
    p.add_argument("--headline", action="append", default=[])
    p.add_argument("--description", action="append", default=[])
    p.add_argument("--keyword", action="append", default=[])
    args = p.parse_args()

    client = build_client()
    customer_id = args.customer.replace('-', '')

    if args.utm_suffix:
        update_campaign_suffix(client, customer_id, args.campaign_id, args.utm_suffix)

    ag_res = create_ad_group(client, customer_id, args.campaign_id, args.adgroup_name)
    add_keywords(client, customer_id, ag_res, args.keyword)

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
    ad_res = add_rsa(client, customer_id, ag_res, args.final_url, heads, descs)
    print("Created ad group:", ag_res)
    print("Created RSA:", ad_res)

if __name__ == "__main__":
    try:
        main()
    except GoogleAdsException as e:
        print("Google Ads API error:", e)
        for err in e.failure.errors:
            print(" -", err)
        raise

