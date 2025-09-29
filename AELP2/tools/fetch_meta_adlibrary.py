#!/usr/bin/env python3
from __future__ import annotations
"""
Meta Ad Library fetcher
  - Reads `competitive/brand_scope.json` for search terms.
  - Calls the Meta Ad Library API (ads_archive) when an access token is
    available.
  - Falls back to cached Aura creative objects when the API is not accessible.

Environment variables:
  META_ADLIBRARY_ACCESS_TOKEN (or META_ACCESS_TOKEN) – required for live calls
  META_ADLIBRARY_COUNTRIES   – comma-separated list (default: "US")
  META_ADLIBRARY_LIMIT       – per-search cap (default: 200)
  META_API_VERSION           – Graph API version (default: v19.0)

Outputs:
  AELP2/competitive/ad_items_raw.json (list of items with metadata)
  AELP2/competitive/ad_items_raw.jsonl (one ad JSON per line)
"""
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / 'AELP2' / 'competitive' / 'brand_scope.json'
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT / 'AELP2' / 'competitive' / 'ad_items_raw.json'
OUT_L = ROOT / 'AELP2' / 'competitive' / 'ad_items_raw.jsonl'

def from_creative_objects():
    items=[]
    for fp in COBJ.glob('*.json'):
        try:
            d=json.loads(fp.read_text())
        except Exception:
            continue
        spec=((d.get('creative') or {}).get('asset_feed_spec')) or {}
        items.append({
            'brand': 'Aura',
            'source': 'repo_meta_cache',
            'title': (spec.get('titles') or [{}])[0].get('text') if spec.get('titles') else None,
            'body': (spec.get('bodies') or [{}])[0].get('text') if spec.get('bodies') else None,
            'link': (spec.get('link_urls') or [{}])[0].get('website_url') if spec.get('link_urls') else None,
            'placements': spec.get('ad_formats') or [],
            'created': (d.get('ad') or {}).get('created_time'),
            'id': (d.get('ad') or {}).get('id')
        })
    return items

def _token() -> str | None:
    return os.getenv('META_ADLIBRARY_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN')


def _countries() -> list[str]:
    raw = os.getenv('META_ADLIBRARY_COUNTRIES', 'US')
    return [c.strip() for c in raw.split(',') if c.strip()]


def _api_version() -> str:
    return os.getenv('META_API_VERSION', 'v19.0')


def _limit_per_term() -> int:
    try:
        return max(1, int(os.getenv('META_ADLIBRARY_LIMIT', '200')))
    except Exception:
        return 200


def fetch_adlibrary(scope: dict) -> list[dict]:
    token = _token()
    if not token:
        return []

    window_days = int(scope.get('window_days', 60))
    since = (datetime.utcnow() - timedelta(days=window_days)).strftime('%Y-%m-%d')
    until = datetime.utcnow().strftime('%Y-%m-%d')
    countries = _countries()
    api_version = _api_version()
    per_term_limit = _limit_per_term()
    # Field names must use Ad Library API plural forms (per official docs):
    # ad_creative_bodies, ad_creative_link_captions, ad_creative_link_descriptions, ad_creative_link_titles
    fields = [
        'id',
        'ad_creation_time',
        'ad_delivery_start_time',
        'ad_delivery_stop_time',
        'ad_creative_bodies',
        'ad_creative_link_captions',
        'ad_creative_link_descriptions',
        'ad_creative_link_titles',
        'ad_snapshot_url',
        'currency',
        'demographic_distribution',
        'delivery_by_region',
        'impressions',
        'publisher_platforms',
        'spend',
        'page_id',
        'page_name',
    ]

    base_url = f'https://graph.facebook.com/{api_version}/ads_archive'
    items: list[dict] = []

    for category, terms in scope.items():
        if category in {'window_days', 'channel'}:
            continue
        if not isinstance(terms, list):
            continue

        for term in terms:
            params = {
                'access_token': token,
                'search_terms': term,
                'ad_active_status': 'ALL',
                'ad_reached_countries': json.dumps(countries),
                'ad_delivery_date_min': since,
                'ad_delivery_date_max': until,
                'limit': min(100, per_term_limit),
                'fields': ','.join(fields),
            }

            fetched = 0
            url = base_url
            while url and fetched < per_term_limit:
                try:
                    resp = requests.get(url, params=params if url == base_url else None, timeout=60)
                except requests.RequestException as exc:
                    print(json.dumps({'warning': f'meta_adlibrary_request_failed', 'term': term, 'error': str(exc)}))
                    break

                if resp.status_code != 200:
                    print(json.dumps({'warning': 'meta_adlibrary_non_200', 'term': term, 'status': resp.status_code, 'body': resp.text[:200]}))
                    break

                data = resp.json()
                rows = data.get('data', [])
                for row in rows:
                    row['_category'] = category
                    row['_search_term'] = term
                    row['_fetched_at'] = datetime.utcnow().isoformat()
                    row['_source'] = 'meta_adlibrary'
                    items.append(row)

                fetched += len(rows)
                next_url = data.get('paging', {}).get('next')
                if not next_url or fetched >= per_term_limit:
                    break
                url = next_url
                params = None  # params already embedded in next URL
                time.sleep(0.5)  # be polite

    return items


def main():
    scope = json.loads(SCOPE.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)

    library_items = fetch_adlibrary(scope)
    fallback_items = from_creative_objects()

    all_items = library_items + fallback_items

    OUT.write_text(json.dumps({
        'fetched': len(library_items),
        'fallback': len(fallback_items),
        'items': all_items,
        'countries': _countries(),
        'note': 'Meta Ad Library + repo cache',
    }, indent=2))

    with OUT_L.open('w') as fh:
        for item in all_items:
            fh.write(json.dumps(item) + '\n')

    print(json.dumps({'count_total': len(all_items), 'count_adlibrary': len(library_items), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()
