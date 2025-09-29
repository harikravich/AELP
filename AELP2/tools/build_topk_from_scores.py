#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / 'AELP2' / 'reports' / 'vendor_scores.json'
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT / 'AELP2' / 'reports' / 'vendor_top20.json'


def load_meta(cid: str) -> dict:
    # Find the first matching creative object file
    for fp in COBJ.glob(f'vendor_*_{cid}.json'):
        try:
            d = json.loads(fp.read_text())
            spec = (d.get('creative') or {}).get('asset_feed_spec') or {}
            title = ''
            bodies = spec.get('bodies') or []
            titles = spec.get('titles') or []
            if titles:
                title = titles[0].get('text') or ''
            body = ''
            if bodies:
                body = bodies[0].get('text') or ''
            link_urls = spec.get('link_urls') or []
            dest = link_urls[0].get('website_url') if link_urls else ''
            page_id = (spec.get('object_story_spec') or {}).get('page_id') or ''
            return {
                'title': title,
                'ad_text': body,
                'destination_url': dest,
                'page_id': page_id,
            }
        except Exception:
            continue
    return {}


def main():
    s = json.loads(SCORES.read_text())
    items = s.get('items') or []
    top = items[:20]
    out_rows = []
    for r in top:
        cid = r.get('creative_id')
        meta = load_meta(cid)
        out_rows.append({
            'creative_id': cid,
            'p_win': r.get('p_win'),
            'lcb': r.get('lcb'),
            **meta,
        })
    OUT.write_text(json.dumps({'count': len(out_rows), 'items': out_rows}, indent=2))
    print(json.dumps({'count': len(out_rows), 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()

