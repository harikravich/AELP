#!/usr/bin/env python3
from __future__ import annotations
"""
Fetch specific Meta ad creatives by ID and download preview images.

Inputs:
  --ids-file artifacts/meta/ads.csv (expects column creative_id) OR --ids comma-separated

Env:
  META_ACCESS_TOKEN

Outputs:
  - assets/meta_creatives/*.jpg
  - assets/meta_creatives/ads_manifest.csv (ad_id=creative_id)
"""
import argparse, csv, os, time, re
from pathlib import Path
from typing import List, Dict, Any
import requests

API_VER = os.getenv('META_API_VERSION', 'v18.0')


def g(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://graph.facebook.com/{API_VER}/{endpoint.lstrip('/')}"
    r = requests.get(base, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Meta API error {r.status_code}: {r.text[:200]}")
    return r.json()


def download(url: str, out_path: Path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)


def get_img_url(cid: str, token: str) -> str | None:
    # Try preview
    try:
        prev = g(f"{cid}/previews", {'access_token': token, 'ad_format': 'DESKTOP_FEED_STANDARD'})
        data = prev.get('data', [])
        if data and 'body' in data[0]:
            html = data[0]['body']
            m = re.search(r'src\s*=\s*"(https?://[^"]+)"', html)
            if m:
                return m.group(1)
    except Exception:
        pass
    # Fallback: creative fields
    try:
        cr = g(cid, {'fields': 'thumbnail_url,image_url,object_story_spec', 'access_token': token})
        for k in ('thumbnail_url','image_url'):
            if isinstance(cr.get(k), str):
                return cr[k]
        oss = cr.get('object_story_spec') or {}
        link = (oss or {}).get('link_data') or {}
        pic = link.get('picture')
        if isinstance(pic, str):
            return pic
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids-file', help='CSV with creative_id column')
    ap.add_argument('--ids', help='Comma-separated creative IDs')
    ap.add_argument('--out-dir', default='assets/meta_creatives')
    ap.add_argument('--manifest', default='assets/meta_creatives/ads_manifest.csv')
    args = ap.parse_args()

    token = os.getenv('META_ACCESS_TOKEN')
    if not token:
        raise SystemExit('Missing META_ACCESS_TOKEN')

    ids: List[str] = []
    if args.ids_file:
        import pandas as pd
        df = pd.read_csv(args.ids_file)
        ids = [str(x) for x in df['creative_id'].dropna().unique().tolist()]
    if args.ids:
        ids.extend([s.strip() for s in args.ids.split(',') if s.strip()])
    ids = sorted(set(ids))
    if not ids:
        raise SystemExit('No creative ids provided')

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in ids:
        url = get_img_url(cid, token)
        asset = ''
        if url:
            asset = str(outdir / f"{cid}.jpg")
            try:
                download(url, Path(asset))
                time.sleep(0.2)
            except Exception:
                asset = ''
        rows.append([cid, asset, '', '', ''])

    man = Path(args.manifest)
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ad_id','asset_path','headline','primary_text','destination_url'])
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} creative assets to {outdir}; manifest {man}")


if __name__ == '__main__':
    main()

