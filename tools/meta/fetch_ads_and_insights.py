#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch Meta Ads list (with creative.id) and daily ad-level insights for a date range.

Env:
  META_ACCESS_TOKEN, META_ACCOUNT_ID (act_...)

Outputs:
  - CSV: artifacts/meta/ads.csv (ad_id, creative_id, name)
  - CSV: artifacts/meta/insights.csv (date, ad_id, creative_id, impressions, clicks, spend)

Usage:
  python tools/meta/fetch_ads_and_insights.py \
    --start 2025-08-01 --end 2025-08-31 --limit 500
"""
import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, List
import requests
import csv


API_VER = os.getenv('META_API_VERSION', 'v18.0')


def g(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://graph.facebook.com/{API_VER}/{endpoint.lstrip('/')}"
    r = requests.get(base, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Meta API error {r.status_code}: {r.text[:200]}")
    return r.json()


def list_ads(account: str, token: str, limit: int = 500) -> List[Dict[str, Any]]:
    ads: List[Dict[str, Any]] = []
    # Include campaign/adset names to enable filtering (e.g., "Balance")
    fields = 'id,name,adset{id,name},campaign{id,name},creative{id,name}'
    url = f"{account}/ads"
    params = {
        'fields': fields,
        'limit': 200,
        'access_token': token
    }
    while True:
        resp = g(url, params)
        data = resp.get('data', [])
        ads.extend(data)
        if len(ads) >= limit:
            break
        paging = resp.get('paging', {})
        next_url = paging.get('next')
        if not next_url:
            break
        # Next page: update URL+params from 'after'
        cursors = paging.get('cursors', {})
        after = cursors.get('after')
        if not after:
            break
        params['after'] = after
        time.sleep(0.2)
    return ads[:limit]


def fetch_insights_for_ads(ad_ids: List[str], token: str, start: str, end: str, date_preset: str | None = None) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    for i in range(0, len(ad_ids), 25):
        batch = ad_ids[i:i+25]
        for ad_id in batch:
            try:
                params = {
                    'access_token': token,
                    'level': 'ad',
                    'time_increment': 1,
                    'fields': 'date_start,impressions,clicks,inline_link_clicks,inline_link_click_ctr,unique_inline_link_clicks,unique_inline_link_click_ctr,spend'
                }
                if date_preset:
                    params['date_preset'] = date_preset
                else:
                    params['time_range'] = f'{{"since":"{start}","until":"{end}"}}'
                resp = g(f"{ad_id}/insights", params)
                for row in resp.get('data', []):
                    insights.append({
                        'date': row.get('date_start'),
                        'ad_id': ad_id,
                        'impressions': int(row.get('impressions', 0) or 0),
                        'clicks': int(row.get('clicks', 0) or 0),
                        'inline_link_clicks': int(row.get('inline_link_clicks', 0) or 0),
                        'inline_link_click_ctr': float(row.get('inline_link_click_ctr', 0.0) or 0.0),
                        'unique_inline_link_clicks': int(row.get('unique_inline_link_clicks', 0) or 0),
                        'unique_inline_link_click_ctr': float(row.get('unique_inline_link_click_ctr', 0.0) or 0.0),
                        'spend': float(row.get('spend', 0.0) or 0.0),
                    })
            except Exception as e:
                # Continue on per-ad errors
                pass
        time.sleep(0.3)
    return insights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--limit', type=int, default=500)
    ap.add_argument('--date-preset', default=None, help='e.g., last_90d, maximum')
    ap.add_argument('--outdir', default='artifacts/meta')
    args = ap.parse_args()

    token = os.getenv('META_ACCESS_TOKEN')
    account = os.getenv('META_ACCOUNT_ID')
    if not token or not account:
        raise SystemExit('Missing META_ACCESS_TOKEN or META_ACCOUNT_ID in environment')

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ads = list_ads(account, token, args.limit)
    ads_path = outdir / 'ads.csv'
    with ads_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id','ad_name','adset_id','adset_name','campaign_id','campaign_name','creative_id','creative_name'])
        for a in ads:
            ad_id = str(a.get('id'))
            ad_name = a.get('name') or ''
            cr = a.get('creative') or {}
            adset = a.get('adset') or {}
            camp = a.get('campaign') or {}
            w.writerow([
                ad_id,
                ad_name,
                str(adset.get('id') or ''), adset.get('name') or '',
                str(camp.get('id') or ''), camp.get('name') or '',
                str(cr.get('id') or ''), cr.get('name') or ''
            ])

    ad_ids = [str(a.get('id')) for a in ads if a.get('id')]
    ins = fetch_insights_for_ads(ad_ids, token, args.start, args.end, args.date_preset)
    ins_path = outdir / 'insights.csv'
    with ins_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['date','ad_id','impressions','clicks','inline_link_clicks','inline_link_click_ctr','unique_inline_link_clicks','unique_inline_link_click_ctr','spend'])
        w.writeheader()
        for r in ins:
            w.writerow(r)
    print(f"Wrote {len(ads)} ads -> {ads_path}; insights rows={len(ins)} -> {ins_path}")


if __name__ == '__main__':
    main()
