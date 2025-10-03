#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch ad-level previews (images) and save under creative_id filenames.

Input: --ads-file artifacts/meta/ads.csv (requires columns: ad_id, creative_id)
Env:   META_ACCESS_TOKEN, META_API_VERSION

Outputs:
  - assets/meta_creatives/*.jpg
  - assets/meta_creatives/ads_manifest.csv (ad_id=creative_id)

Usage:
  python tools/meta/fetch_ad_previews.py \
    --ads-file artifacts/meta/ads.csv \
    --out-dir assets/meta_creatives \
    --manifest assets/meta_creatives/ads_manifest.csv
"""
import argparse, os, re, time, csv
from pathlib import Path
from typing import Dict, Any
import requests

API_VER = os.getenv('META_API_VERSION', 'v18.0')


def g(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://graph.facebook.com/{API_VER}/{endpoint.lstrip('/')}"
    r = requests.get(base, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Meta API error {r.status_code}: {r.text[:200]}")
    return r.json()


def extract_img_from_html(html: str) -> str | None:
    m = re.search(r'src\s*=\s*"(https?://[^"\\]+)"', html)
    return m.group(1) if m else None


def download(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ads-file', required=True)
    ap.add_argument('--out-dir', default='assets/meta_creatives')
    ap.add_argument('--manifest', default='assets/meta_creatives/ads_manifest.csv')
    args = ap.parse_args()

    token = os.getenv('META_ACCESS_TOKEN')
    if not token:
        raise SystemExit('Missing META_ACCESS_TOKEN')

    import pandas as pd
    ads = pd.read_csv(args.ads_file)
    need_cols = {'ad_id','creative_id'}
    if not need_cols.issubset(set(ads.columns)):
        raise SystemExit('ads-file must have ad_id and creative_id columns')

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for _, r in ads.iterrows():
        ad_id = str(r['ad_id'])
        cid = str(r['creative_id'])
        img_url = None
        # 1) Try creative fields via ad -> creative
        try:
            info = g(ad_id, {'fields': 'creative{id,thumbnail_url,image_url,object_story_spec}', 'access_token': token})
            cr = (info.get('creative') or {})
            for k in ('thumbnail_url','image_url'):
                if isinstance(cr.get(k), str) and cr[k].startswith('http'):
                    img_url = cr[k]
                    break
        except Exception:
            pass
        # 2) Fallback to ad previews HTML scrape
        if not img_url:
            try:
                prev = g(f"{ad_id}/previews", {'access_token': token, 'ad_format': 'DESKTOP_FEED_STANDARD'})
                data = prev.get('data', [])
                if data and 'body' in data[0]:
                    img_url = extract_img_from_html(data[0]['body'])
            except Exception:
                pass
        asset = ''
        if img_url:
            asset = str(outdir / f"{cid}.jpg")
            try:
                download(img_url, Path(asset))
                time.sleep(0.1)
            except Exception:
                asset = ''
        rows.append([cid, asset, '', '', ''])

    man = Path(args.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id','asset_path','headline','primary_text','destination_url'])
        for row in rows:
            w.writerow(row)
    print(f"Wrote {len(rows)} preview assets to {outdir}; manifest {man}")


if __name__ == '__main__':
    main()
