#!/usr/bin/env python3
from __future__ import annotations
"""
Recompute weekly labels from creative_enriched/*.json and write to creative_weekly/*.json
Labels use the same quantile rules but apply to the weekly-aggregated fields already present.
"""
import json, os, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
DST = ROOT / 'AELP2' / 'reports' / 'creative_weekly'
DST.mkdir(parents=True, exist_ok=True)

def quantile(vals, q):
    if not vals: return float('nan')
    s=sorted(vals); k=max(0,min(len(s)-1,int(q*(len(s)-1))))
    return float(s[k])

def main():
    pct_purch=float(os.getenv('AELP2_PCT_PURCH_POS','0.60'))
    pct_cac=float(os.getenv('AELP2_PCT_CAC_POS','0.40'))
    for f in sorted(SRC.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        purch_vals=[float(it.get('actual_score') or 0.0) for it in items]
        cac_vals=[float(it.get('actual_cac')) for it in items if (it.get('actual_cac') is not None and math.isfinite(it.get('actual_cac')))]
        q_p=quantile(purch_vals, pct_purch)
        q_c=quantile(cac_vals, pct_cac)
        out_items=[]
        for it in items:
            lbl=1 if ((float(it.get('actual_score') or 0.0)>=q_p) and (float(it.get('actual_cac') or float('inf'))<=q_c)) else 0
            it2=dict(it); it2['actual_label']=lbl; out_items.append(it2)
        (DST/f.name).write_text(json.dumps({'campaign_id': d.get('campaign_id'), 'items': out_items}, indent=2))
    print(f'wrote weekly to {DST}')

if __name__=='__main__':
    main()

