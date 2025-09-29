#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch Meta Ad Library results via SearchAPI.io (self-serve) and write CSVs that our
vendor importer understands. Then you can run the normal pipeline to build features
and score creatives.

Env:
  SEARCHAPI_API_KEY  (required)

Usage:
  source .env
  python3 AELP2/tools/fetch_searchapi_meta.py \
    --filters AELP2/config/bigspy_filters.yaml \
    --countries US,GB \
    --days 365 \
    --max-per-query 250 \
    --outdir AELP2/vendor_imports

Notes:
  - Prioritizes EU/UK coverage when you include EU countries (API returns all ad types).
  - US coverage via Ad Library API is limited to political/issue; include EU for breadth.
  - This tool batches your semicolon-separated chips as individual queries.
"""
import argparse, csv, json, os, time, urllib.parse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILTERS = ROOT / 'AELP2' / 'config' / 'bigspy_filters.yaml'
OUTDIR = ROOT / 'AELP2' / 'vendor_imports'


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def split_chips(s: str) -> list[str]:
    chips = [c.strip() for c in (s or '').split(';')]
    return [c for c in chips if c]

def split_excludes(s: str) -> list[str]:
    ex = [c.strip() for c in (s or '').split(';')]
    return [c for c in ex if c]


def http_get(url: str, params: dict[str, Any]) -> dict:
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    return r.json()


def normalize_row(ad: dict) -> dict:
    # Try both flattened and nested structures from SearchAPI.io
    snap = ad.get('snapshot') or {}
    ad_id = ad.get('ad_archive_id') or ad.get('archive_id') or ad.get('id')
    page_id = ad.get('page_id') or (ad.get('page') or {}).get('id')
    title = (
        ad.get('ad_creative_link_title')
        or ad.get('ad_creative_link_caption')
        or snap.get('title')
        or snap.get('caption')
        or ''
    )
    body = ad.get('ad_creative_body') or snap.get('body') or snap.get('message') or ''
    link = (
        ad.get('ad_creative_link_url')
        or snap.get('link_url')
        or snap.get('link_destination_display_url')
        or ''
    )
    platforms = ad.get('publisher_platforms') or ad.get('platforms') or []
    media_type = ad.get('media_type') or (snap.get('cards') and 'video' in json.dumps(snap.get('cards'), ensure_ascii=False).lower() and 'video' or '')

    row = {
        'ad_archive_id': ad_id or '',
        'page_id': page_id or '',
        'title': title or '',
        'ad_text': body or '',
        'destination_url': link or '',
        'platform': ','.join(platforms) if isinstance(platforms, list) else (platforms or ''),
        'media_type': media_type or '',
        'first_seen': ad.get('ad_delivery_start_time') or ad.get('start_date') or '',
        'last_seen': ad.get('ad_delivery_stop_time') or ad.get('end_date') or '',
        'snapshot_url': ad.get('ad_snapshot_url') or snap.get('ad_snapshot_url') or '',
        'page_name': (ad.get('page_name') or (ad.get('page') or {}).get('name') or ''),
        'country': ','.join(ad.get('ad_reached_countries') or [])
    }
    return row


def fetch_queries(chips: list[str], countries: list[str], days: int, max_per_query: int, api_key: str, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    start = date.today() - timedelta(days=days)
    # SearchAPI requires end_date not be in the future; use yesterday to be safe
    end = date.today() - timedelta(days=1)
    # Correct endpoint per SearchAPI docs: use /search with engine=meta_ad_library
    base = 'https://www.searchapi.io/api/v1/search'
    files: list[Path] = []

    for country in countries:
        for chip in chips:
            q = chip.strip()
            # Build exclude tail like: -novel -"short drama" -wattpad
            excludes_raw = os.environ.get('SEARCHAPI_EXCLUDES', '')
            excludes = split_excludes(excludes_raw)
            # Also accept excludes from filters YAML under key 'exclude'
            # Note: we read it outside; pass via env to keep signature simple
            tail = ''
            if excludes:
                parts = []
                for t in excludes:
                    if ' ' in t:
                        parts.append(f'-"{t}"')
                    else:
                        parts.append(f'-{t}')
                tail = ' ' + ' '.join(parts)
            q_with_ex = (f'"{q}"' if ' ' in q else q) + tail
            params = {
                'engine': 'meta_ad_library',
                'q': q_with_ex,
                'country': country,
                'platforms': 'facebook,instagram',
                'start_date': start.isoformat(),
                'end_date': end.isoformat(),
                'languages': 'en',
                'api_key': api_key,
            }
            out = outdir / f"searchapi_meta_{country}_{urllib.parse.quote_plus(q)[:32]}_{end.strftime('%Y%m%d')}.csv"
            seen: set[str] = set()
            total = 0
            next_token = None

            with out.open('w', newline='', encoding='utf-8') as fcsv:
                cols = [
                    'ad_archive_id','page_id','page_name','title','ad_text','destination_url',
                    'platform','media_type','first_seen','last_seen','snapshot_url','country'
                ]
                w = csv.DictWriter(fcsv, fieldnames=cols)
                w.writeheader()

                while True:
                    p = dict(params)
                    if next_token:
                        p['next_page_token'] = next_token
                    data = http_get(base, p)
                    ads = data.get('ads') or data.get('data') or []
                    for ad in ads:
                        r = normalize_row(ad)
                        key = r.get('ad_archive_id')
                        if not key or key in seen:
                            continue
                        seen.add(key)
                        w.writerow(r)
                        total += 1
                        if total >= max_per_query:
                            break
                    if total >= max_per_query:
                        break
                    next_token = (data.get('pagination') or {}).get('next_page_token') or data.get('next_page_token')
                    if not next_token:
                        break
                    time.sleep(0.6)  # polite throttle

            print(json.dumps({'country': country, 'query': q, 'written': total, 'out': str(out)}, indent=2))
            files.append(out)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--filters', default=str(DEFAULT_FILTERS), help='YAML with query, country, language, etc.')
    ap.add_argument('--countries', default='GB', help='Comma-separated list of ISO country codes (e.g., US,GB,DE)')
    ap.add_argument('--days', type=int, default=365, help='Lookback window in days')
    ap.add_argument('--max-per-query', type=int, default=250, help='Max rows to write per chip per country')
    ap.add_argument('--outdir', default=str(OUTDIR))
    args = ap.parse_args()

    api_key = os.environ.get('SEARCHAPI_API_KEY') or ''
    if not api_key:
        raise SystemExit('Missing SEARCHAPI_API_KEY in environment. Sign up at searchapi.io and add to .env.')

    filters = load_yaml(Path(args.filters)) if args.filters else {}
    chips = split_chips(filters.get('query') or '')
    if not chips:
        raise SystemExit('No chips found in filters YAML under key "query" (semicolon-separated).')
    countries = [c.strip().upper() for c in args.countries.split(',') if c.strip()]
    outdir = Path(args.outdir)

    files = fetch_queries(chips, countries, args.days, args.max_per_query, api_key, outdir)
    print(json.dumps({'files': [str(f) for f in files]}, indent=2))


if __name__ == '__main__':
    main()
