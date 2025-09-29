#!/usr/bin/env python3
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
COBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT / 'AELP2' / 'reports' / 'copy_bank.json'

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or '').strip())

def main():
    titles=Counter(); bodies=Counter()
    for f in sorted(COBJ.glob('*.json')):
        try:
            d=json.loads(f.read_text())
        except Exception:
            continue
        # top-level title/body if present
        t = (((d.get('creative') or {}).get('title')) or '') if isinstance((d.get('creative') or {}).get('title'), str) else ''
        b = (((d.get('creative') or {}).get('body')) or '') if isinstance((d.get('creative') or {}).get('body'), str) else ''
        if t: titles[norm(t)] += 1
        if b: bodies[norm(b)] += 1
        # asset feeds
        af = ((d.get('creative') or {}).get('asset_feed_spec') or {})
        for arr_key in ('titles','bodies'):
            for item in (af.get(arr_key) or []):
                txt = norm(item.get('text') or '')
                if not txt: continue
                if arr_key=='titles': titles[txt]+=1
                else: bodies[txt]+=1
    out={
        'titles': [{'text': k, 'count': v} for k,v in titles.most_common(200)],
        'bodies': [{'text': k, 'count': v} for k,v in bodies.most_common(200)]
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'titles': len(out['titles']), 'bodies': len(out['bodies'])}, indent=2))

if __name__=='__main__':
    main()

