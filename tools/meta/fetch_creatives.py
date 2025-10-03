#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch Meta (Facebook/Instagram) ad creatives and download preview images.

Requires env vars: META_ACCESS_TOKEN, META_ACCOUNT_ID (act_...)

Outputs:
  - Downloads preview images to assets/meta_creatives/
  - Writes CSV manifest with columns: ad_id,asset_path,headline,primary_text,destination_url

Usage:
  META_ACCESS_TOKEN=... META_ACCOUNT_ID=act_... \
  python tools/meta/fetch_creatives.py --limit 50 \
    --out-dir assets/meta_creatives \
    --manifest assets/meta_creatives/ads_manifest.csv
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import json
import urllib.parse

import requests


API_VER = os.getenv('META_API_VERSION', 'v18.0')


def g(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://graph.facebook.com/{API_VER}/{endpoint.lstrip('/')}"
    r = requests.get(base, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Meta API error {r.status_code}: {r.text[:200]}")
    return r.json()


def get_image_url_for_creative(creative: Dict[str, Any], token: str) -> Optional[str]:
    # Try thumbnail_url or image_url first
    for k in ('thumbnail_url', 'image_url'):
        url = creative.get(k)
        if isinstance(url, str) and url.startswith('http'):
            return url
    # Try object_story_spec -> link_data.picture
    oss = creative.get('object_story_spec') or {}
    link = (oss or {}).get('link_data') or {}
    pic = link.get('picture')
    if isinstance(pic, str) and pic.startswith('http'):
        return pic
    # Fallback: use previews endpoint to get an embeddable HTML and scrape image URL
    try:
        prev = g(f"{creative['id']}/previews", {
            'access_token': token,
            'ad_format': 'DESKTOP_FEED_STANDARD'
        })
        data = prev.get('data', [])
        if data and 'body' in data[0]:
            html = data[0]['body']
            # naive extract of src="..." for first image
            import re
            m = re.search(r'src\s*=\s*"(https?://[^"]+)"', html)
            if m:
                return m.group(1)
    except Exception:
        pass
    return None


def download(url: str, out_path: Path):
    # Some URLs require URL-encoding of query
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
    except Exception:
        # Retry with encoded URL
        parts = urllib.parse.urlsplit(url)
        url2 = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, urllib.parse.quote_plus(parts.query), parts.fragment))
        r = requests.get(url2, timeout=60)
        r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('wb') as f:
        f.write(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=50)
    ap.add_argument('--out-dir', default='assets/meta_creatives')
    ap.add_argument('--manifest', default='assets/meta_creatives/ads_manifest.csv')
    args = ap.parse_args()

    token = os.getenv('META_ACCESS_TOKEN')
    account = os.getenv('META_ACCOUNT_ID')
    if not token or not account:
        raise SystemExit('Missing META_ACCESS_TOKEN or META_ACCOUNT_ID in environment')

    fields = ','.join([
        'id', 'name', 'object_story_spec', 'thumbnail_url', 'image_url'
    ])
    # Page through adcreatives until reaching desired limit
    data: List[Dict[str, Any]] = []
    params: Dict[str, Any] = {
        'fields': fields,
        'limit': 100,
        'access_token': token,
    }
    next_params = params.copy()
    while True:
        try:
            resp = g(f"{account}/adcreatives", next_params)
        except Exception:
            # Backoff to smaller page size on API stress
            next_params['limit'] = max(25, int(next_params.get('limit', 100)) // 2)
            resp = g(f"{account}/adcreatives", next_params)
        page = resp.get('data', [])
        data.extend(page)
        if len(data) >= args.limit:
            break
        paging = resp.get('paging', {})
        cursors = paging.get('cursors', {}) if isinstance(paging, dict) else {}
        after = cursors.get('after') if isinstance(cursors, dict) else None
        if not after:
            break
        next_params['after'] = after
        # keep page size modest
        next_params['limit'] = min(100, next_params.get('limit', 100))
        time.sleep(0.2)
    data = data[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[List[str]] = []
    for it in data:
        cid = str(it.get('id'))
        name = it.get('name') or cid
        oss = it.get('object_story_spec') or {}
        link = (oss or {}).get('link_data') or {}
        headline = (link.get('name') or '').strip()
        primary = (link.get('message') or '').strip()
        dest = (link.get('link') or '').strip()

        img_url = get_image_url_for_creative(it, token)
        asset_path = ''
        if img_url:
            fn = f"{cid}.jpg"
            asset_path = str(out_dir / fn)
            try:
                download(img_url, Path(asset_path))
                time.sleep(0.2)  # polite
            except Exception as e:
                # Skip download errors but keep metadata
                asset_path = ''
        manifest_rows.append([cid, asset_path, headline, primary, dest])

    man = Path(args.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id', 'asset_path', 'headline', 'primary_text', 'destination_url'])
        for row in manifest_rows:
            w.writerow(row)
    print(f"Wrote manifest for {len(manifest_rows)} creatives to {man}")


if __name__ == '__main__':
    main()
