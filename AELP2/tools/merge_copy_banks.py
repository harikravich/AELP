#!/usr/bin/env python3
from __future__ import annotations
"""
Merge copy from Meta (copy_bank.json), Google Ads (google_ads_copy.json), Impact (impact_copy.json), and YouTube (youtube_copy.json).
Outputs AELP2/reports/copy_bank_merged.json with deduped lines and simple tags per source.
"""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'AELP2' / 'reports' / 'copy_bank.json'
GADS = ROOT / 'AELP2' / 'reports' / 'google_ads_copy.json'
IMPA = ROOT / 'AELP2' / 'reports' / 'impact_copy.json'
YT   = ROOT / 'AELP2' / 'reports' / 'youtube_copy.json'
OUT  = ROOT / 'AELP2' / 'reports' / 'copy_bank_merged.json'

def load_json(p):
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except Exception: return None

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or '').strip())

def main():
    meta=load_json(META) or {}
    gads=load_json(GADS) or {}
    impa=load_json(IMPA) or {}
    yt=load_json(YT) or {}
    lines={}
    def add(txt, src):
        t=norm(txt)
        if len(t)<6: return
        rec=lines.get(t) or {'text': t, 'sources': set(), 'count': 0}
        rec['sources'].add(src); rec['count']+=1
        lines[t]=rec
    # meta
    for k in ('titles','bodies'):
        for it in (meta.get(k) or []): add(it.get('text') or '', f'meta:{k}')
    # google ads
    for k in ('headlines','descriptions'):
        for it in (gads.get(k) or []): add(it.get('text') or '', f'google_ads:{k}')
    # impact
    for it in (impa.get('items') or []): add(it.get('text') or '', f'impact:{it.get("field")}')
    # youtube (titles only)
    for it in (yt.get('items') or []): add(it.get('title') or '', 'youtube:title')
    merged=sorted([
        {'text': v['text'], 'sources': sorted(list(v['sources'])), 'count': v['count']} for v in lines.values()
    ], key=lambda x: -x['count'])
    OUT.write_text(json.dumps({'items': merged[:1000], 'summary': {'total': len(merged)}}, indent=2))
    print(json.dumps({'total': len(merged)}, indent=2))

if __name__=='__main__':
    main()

