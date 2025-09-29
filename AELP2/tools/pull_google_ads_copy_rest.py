#!/usr/bin/env python3
from __future__ import annotations
"""
Pull Google Ads headlines/descriptions via REST (searchStream) using OAuth refresh token.
Avoids gRPC compatibility issues.

Env required:
- GOOGLE_ADS_DEVELOPER_TOKEN
- GOOGLE_ADS_CLIENT_ID
- GOOGLE_ADS_CLIENT_SECRET
- GOOGLE_ADS_REFRESH_TOKEN
- GOOGLE_ADS_CUSTOMER_ID (child; 10 digits)
- (optional) GOOGLE_ADS_LOGIN_CUSTOMER_ID (MCC)

Output: AELP2/reports/google_ads_copy.json
"""
import os, json, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'AELP2' / 'reports' / 'google_ads_copy.json'

def get_access_token(client_id, client_secret, refresh_token):
    url = 'https://oauth2.googleapis.com/token'
    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }
    r = requests.post(url, data=data, timeout=30)
    r.raise_for_status()
    return r.json()['access_token']

def search_stream(access_token, developer_token, customer_id, login_customer_id, query):
    versions = ['v18','v17','v16']
    headers = {
        'Authorization': f'Bearer {access_token}',
        'developer-token': developer_token,
        'Content-Type': 'application/json'
    }
    if login_customer_id and login_customer_id != customer_id:
        headers['login-customer-id'] = login_customer_id
    last_err=None
    for ver in versions:
        url = f'https://googleads.googleapis.com/{ver}/customers/{customer_id}/googleAds:search'
        r = requests.post(url, headers=headers, json={'query': query}, timeout=60)
        if r.status_code < 400:
            return r.json(), None
        last_err={'status_code': r.status_code, 'body': r.text[:500], 'version': ver}
    return None, last_err

def main():
    try:
        dev = os.environ['GOOGLE_ADS_DEVELOPER_TOKEN']
        cid = os.environ['GOOGLE_ADS_CLIENT_ID']
        csec = os.environ['GOOGLE_ADS_CLIENT_SECRET']
        rtok = os.environ['GOOGLE_ADS_REFRESH_TOKEN']
        cust = os.environ['GOOGLE_ADS_CUSTOMER_ID']
        login = os.environ.get('GOOGLE_ADS_LOGIN_CUSTOMER_ID')
    except KeyError as e:
        OUT.write_text(json.dumps({'error': f'missing env {e.args[0]}'}, indent=2))
        print(json.dumps({'status': 'error', 'detail': f'missing env {e.args[0]}'}, indent=2))
        return

    try:
        token = get_access_token(cid, csec, rtok)
    except Exception as ex:
        OUT.write_text(json.dumps({'error': f'oauth_failed: {ex}'}, indent=2))
        print(json.dumps({'status': 'error', 'detail': 'oauth_failed'}, indent=2))
        return

    gaql = (
        "SELECT ad_group_ad.ad.id, ad_group_ad.ad.type, ad_group_ad.status, "
        "ad_group_ad.ad.responsive_search_ad.headlines, ad_group_ad.ad.responsive_search_ad.descriptions, "
        "ad_group_ad.ad.expanded_text_ad.headline_part1, ad_group_ad.ad.expanded_text_ad.headline_part2, "
        "ad_group_ad.ad.expanded_text_ad.description "
        "FROM ad_group_ad WHERE ad_group_ad.status != 'REMOVED' LIMIT 2000"
    )
    js, err = search_stream(token, dev, cust, login, gaql)
    if err:
        OUT.write_text(json.dumps({'error': err}, indent=2))
        print(json.dumps({'status': 'error', 'detail': 'rest_error'}, indent=2))
        return

    headlines = {}
    descriptions = {}
    for batch in js or []:
        for row in batch.get('results', []):
            ad = ((row.get('adGroupAd') or {}).get('ad') or {})
            ad_type = (ad.get('type') or {}).get('name') or ad.get('type')
            if 'responsiveSearchAd' in ad:
                rsa = ad['responsiveSearchAd']
                for h in rsa.get('headlines', []):
                    t = (h.get('text') or '').strip()
                    if t: headlines[t] = headlines.get(t, 0) + 1
                for d in rsa.get('descriptions', []):
                    t = (d.get('text') or '').strip()
                    if t: descriptions[t] = descriptions.get(t, 0) + 1
            if 'expandedTextAd' in ad:
                eta = ad['expandedTextAd']
                for t in [eta.get('headlinePart1'), eta.get('headlinePart2')]:
                    if t: headlines[t.strip()] = headlines.get(t.strip(), 0) + 1
                if eta.get('description'):
                    descriptions[eta['description'].strip()] = descriptions.get(eta['description'].strip(), 0) + 1

    out = {
        'headlines': sorted([{'text': k, 'count': v} for k,v in headlines.items()], key=lambda x: -x['count'])[:500],
        'descriptions': sorted([{'text': k, 'count': v} for k,v in descriptions.items()], key=lambda x: -x['count'])[:500]
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'headlines': len(out['headlines']), 'descriptions': len(out['descriptions'])}, indent=2))

if __name__=='__main__':
    main()
