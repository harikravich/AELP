#!/usr/bin/env python3
"""
Fetch Meta Ad and AdCreative objects for historical ads listed in AELP2/reports/creative/*.json.
Outputs one JSON per ad under AELP2/reports/creative_objects/<ad_id>.json including:
  - ad: id, name, created_time, effective_status, creative{id}
  - creative: id, object_story_spec, asset_feed_spec, body/title/link_url if present

Offline-only: read-only Graph calls.
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
from typing import Dict
import requests

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / 'AELP2' / 'reports' / 'creative'
OUT = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT.mkdir(parents=True, exist_ok=True)
META_BASE = os.getenv('META_BASE_URL', 'https://graph.facebook.com/v23.0')


def read_env_token() -> str:
    tok = os.getenv('META_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN_DISABLED')
    if not tok and (ROOT / '.env').exists():
        for ln in (ROOT / '.env').read_text().splitlines():
            if ln.startswith('export META_ACCESS_TOKEN='):
                tok = ln.split('=',1)[1].strip()
                break
            if ln.startswith('export META_ACCESS_TOKEN_DISABLED=') and not tok:
                tok = ln.split('=',1)[1].strip()
    if not tok:
        raise RuntimeError('Missing META_ACCESS_TOKEN')
    return tok


def fetch_json(url: str, params: Dict[str,str], retries: int = 3) -> dict:
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (i+1))
            continue
        r.raise_for_status()
    r.raise_for_status()


def main():
    token = read_env_token()
    ad_ids = set()
    for f in sorted(INP.glob('*.json')):
        data = json.loads(f.read_text())
        for it in (data.get('items') or []):
            ad_ids.add(it.get('creative_id'))
    print(f'Found {len(ad_ids)} ad IDs')

    for ad_id in sorted(ad_ids):
        if not ad_id:
            continue
        outp = OUT / f'{ad_id}.json'
        if outp.exists():
            continue
        # Fetch Ad object
        ad_url = f"{META_BASE}/{ad_id}"
        ad = fetch_json(ad_url, params={'fields': 'id,name,created_time,effective_status,creative', 'access_token': token})
        creative = {}
        try:
            cid = ((ad.get('creative') or {}).get('id'))
            if cid:
                cr_url = f"{META_BASE}/{cid}"
                creative = fetch_json(cr_url, params={'fields': 'id,name,object_story_spec,asset_feed_spec,title,body,link_url', 'access_token': token})
        except Exception:
            pass
        outp.write_text(json.dumps({'ad': ad, 'creative': creative}, indent=2))
        time.sleep(0.1)
    print('Done')


if __name__ == '__main__':
    main()

