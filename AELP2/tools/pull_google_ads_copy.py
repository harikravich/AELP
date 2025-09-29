#!/usr/bin/env python3
from __future__ import annotations
"""
Pull Google Ads headlines/descriptions via the Google Ads API (read-only) using env credentials.
Writes AELP2/reports/google_ads_copy.json with top texts and counts.

Env required:
- GOOGLE_ADS_DEVELOPER_TOKEN
- GOOGLE_ADS_CLIENT_ID
- GOOGLE_ADS_CLIENT_SECRET
- GOOGLE_ADS_REFRESH_TOKEN
- GOOGLE_ADS_CUSTOMER_ID (child; 10 digits)
- (optional) GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)
"""
import os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'google_ads_copy.json'

def load_config():
    cfg = {
        'developer_token': os.environ['GOOGLE_ADS_DEVELOPER_TOKEN'],
        'client_id': os.environ['GOOGLE_ADS_CLIENT_ID'],
        'client_secret': os.environ['GOOGLE_ADS_CLIENT_SECRET'],
        'refresh_token': os.environ['GOOGLE_ADS_REFRESH_TOKEN'],
        'login_customer_id': os.environ.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID'),
        'customer_id': os.environ['GOOGLE_ADS_CUSTOMER_ID']
    }
    return cfg

def main():
    cfg = load_config()
    try:
        from google.ads.googleads.client import GoogleAdsClient
        from google.ads.googleads.errors import GoogleAdsException
    except Exception as e:
        OUT.write_text(json.dumps({'error': f'missing google-ads library: {e}'}, indent=2))
        print(json.dumps({'status': 'error', 'detail': 'missing google-ads'}, indent=2))
        return

    # Build in-memory config
    ads_config = {
        'developer_token': cfg['developer_token'],
        'client_id': cfg['client_id'],
        'client_secret': cfg['client_secret'],
        'refresh_token': cfg['refresh_token'],
        'use_proto_plus': True,
        'login_customer_id': cfg.get('login_customer_id')
    }
    client = GoogleAdsClient.load_from_dict(ads_config)
    ga_service = client.get_service('GoogleAdsService')
    customer_id = cfg['customer_id']

    # Query RSA and ETA texts
    query = (
        "SELECT ad_group_ad.ad.id, ad_group_ad.ad.type, ad_group_ad.status, "
        "ad_group_ad.ad.responsive_search_ad.headlines, ad_group_ad.ad.responsive_search_ad.descriptions, "
        "ad_group_ad.ad.expanded_text_ad.headline_part1, ad_group_ad.ad.expanded_text_ad.headline_part2, "
        "ad_group_ad.ad.expanded_text_ad.description "
        "FROM ad_group_ad WHERE ad_group_ad.status != 'REMOVED' LIMIT 2000"
    )
    headlines = {}
    descriptions = {}
    try:
        stream = ga_service.search_stream(customer_id=customer_id, query=query)
        for batch in stream:
            for row in batch.results:
                ad = row.ad_group_ad.ad
                if ad.type_.name == 'RESPONSIVE_SEARCH_AD':
                    for a in ad.responsive_search_ad.headlines:
                        txt = (a.text or '').strip()
                        if txt:
                            headlines[txt] = headlines.get(txt, 0) + 1
                    for a in ad.responsive_search_ad.descriptions:
                        txt = (a.text or '').strip()
                        if txt:
                            descriptions[txt] = descriptions.get(txt, 0) + 1
                elif ad.type_.name == 'EXPANDED_TEXT_AD':
                    for txt in [ad.expanded_text_ad.headline_part1, ad.expanded_text_ad.headline_part2]:
                        if txt:
                            headlines[txt] = headlines.get(txt, 0) + 1
                    if ad.expanded_text_ad.description:
                        descriptions[ad.expanded_text_ad.description] = descriptions.get(ad.expanded_text_ad.description, 0) + 1
    except GoogleAdsException as ex:
        OUT.write_text(json.dumps({'error': 'GoogleAdsException', 'message': ex.failure.message}, indent=2))
        print(json.dumps({'status': 'error', 'detail': 'GoogleAdsException'}, indent=2))
        return
    except Exception as ex:
        OUT.write_text(json.dumps({'error': str(ex)}, indent=2))
        print(json.dumps({'status': 'error', 'detail': str(ex)}, indent=2))
        return

    out = {
        'headlines': sorted([{'text': k, 'count': v} for k,v in headlines.items()], key=lambda x: -x['count'])[:500],
        'descriptions': sorted([{'text': k, 'count': v} for k,v in descriptions.items()], key=lambda x: -x['count'])[:500]
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'headlines': len(out['headlines']), 'descriptions': len(out['descriptions'])}, indent=2))

if __name__=='__main__':
    main()

