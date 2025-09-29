#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'

def actual_cac(it: dict) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return (spend / purch) if purch > 0 else float('inf')

def pick_baseline(items):
    def good_volume(it):
        return (float(it.get('actual_score') or 0.0) >= 2) or (float(it.get('test_clicks') or 0.0) >= 50)
    eligible = [it for it in items if good_volume(it)]
    if not eligible:
        return None
    eligible.sort(key=lambda it: (actual_cac(it), -float(it.get('actual_score') or 0.0)))
    return eligible[0]

def utility(it: dict, target_cac: float) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return purch - (spend/target_cac if target_cac>0 else 0.0)

def impressions(items):
    imps=[]
    for it in items:
        stats = it.get('placement_stats') or {}
        imps.append(sum((v or {}).get('impr',0.0) for v in stats.values()))
    return imps

def snips(values, props):
    numer=0.0; denom=0.0
    for v,p in zip(values, props):
        if p<=0: continue
        w=1.0/p
        numer+=w*v; denom+=w
    return numer/denom if denom>0 else 0.0

def main():
    files = sorted(ENR.glob('*.json'))
    per=[]
    for f in files:
        d=json.loads(f.read_text()); items=d.get('items') or []
        if not items:
            continue
        base = pick_baseline(items)
        if not base:
            continue
        # Target CAC = campaign median
        cvals = [actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
        target_cac = float(sorted(cvals)[len(cvals)//2]) if cvals else 30.0
        b_sim = float(base.get('sim_score') or 0.0)
        # Score uplift as sim_score - base_sim
        scored = []
        for it in items:
            if it is base: continue
            scored.append((float(it.get('sim_score') or 0.0) - b_sim, it))
        scored.sort(key=lambda x: -x[0])
        topk = [it for _,it in scored[:5]]
        # Compute SNIPS for uplift utility: U(variant)-U(base)
        values=[]; props=[]
        imps = impressions(items)
        imp_sum = sum(imps) or 1.0
        imp_map = {items[i].get('creative_id'): (imps[i]/imp_sum) for i in range(len(items))}
        for it in topk:
            u = utility(it, target_cac) - utility(base, target_cac)
            p = float(imp_map.get(it.get('creative_id'), 1e-6))
            values.append(u); props.append(p)
        est = snips(values, props) if values else 0.0
        per.append({'campaign_file': f.name, 'uplift_snips_top5': est, 'baseline': base.get('creative_id')})
    OUT.joinpath('ope_uplift_topk.json').write_text(json.dumps({'results': per}, indent=2))
    print(json.dumps({'campaigns': len(per)}, indent=2))

if __name__=='__main__':
    main()

