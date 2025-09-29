#!/usr/bin/env python3
from __future__ import annotations
"""
Baseline utility selector: rank creatives by expected utility proxy and export Top-N per campaign (offline only).
Utility: U = purchases - spend/target_CAC, with target from target_cac.json or campaign median.
Outputs: AELP2/reports/utility_topn.json
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0); return (s/p) if p>0 else float('inf')

def utility(it: dict, target: float) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0); return p - (s/target if target>0 else 0.0)

def main():
    N=int(os.getenv('AELP2_TOPN','10'))
    tgtp=ROOT/'AELP2/reports/target_cac.json'
    tgt=json.loads(tgtp.read_text()) if tgtp.exists() else {}
    out=[]
    for f in sorted(ENR.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        if not items: continue
        key=f.stem
        tc=tgt.get(key,{}).get('target_cac')
        if not tc:
            cacs=[actual_cac(it) for it in items if actual_cac(it)<1e6]
            tc = sorted(cacs)[len(cacs)//2] if cacs else 200.0
        ranked=sorted(items, key=lambda it: utility(it, tc), reverse=True)[:N]
        out.append({'campaign_file': f.name, 'target_cac': tc, 'topN': [{'creative_id': it.get('creative_id'), 'U': utility(it, tc)} for it in ranked]})
    (OUT/'utility_topn.json').write_text(json.dumps({'results': out}, indent=2))
    print('AELP2/reports/utility_topn.json')

if __name__=='__main__':
    main()

