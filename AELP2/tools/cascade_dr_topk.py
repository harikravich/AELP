#!/usr/bin/env python3
from __future__ import annotations
"""
Cascade-DR style estimator for a Top-K slate policy.
Simplified examination model: prob(examine rank r) = eta^(r-1), eta estimated from CTR drop-off.
"""
import json, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import os
ENR = Path(os.getenv('AELP2_CREATIVE_DIR') or (ROOT / 'AELP2' / 'reports' / 'creative_enriched'))
OUT = ROOT / 'AELP2' / 'reports'

def estimate_eta(items):
    # approximate CTRs from clicks/impr in placement_stats
    ctrs=[]
    for it in items:
        stats = it.get('placement_stats') or {}
        impr = sum((v or {}).get('impr',0.0) for v in stats.values())
        clk = sum((v or {}).get('clicks',0.0) for v in stats.values())
        if impr>0: ctrs.append(clk/impr)
    if len(ctrs)<2: return 0.8
    ctrs = sorted(ctrs, reverse=True)
    # ratio of second to first as crude eta
    eta = ctrs[1]/ctrs[0] if ctrs[0]>0 else 0.8
    return float(np.clip(eta, 0.5, 0.95))

def main():
    k = int(os.getenv('AELP2_OPE_TOPK', '5'))
    res=[]
    for f in sorted(ENR.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        if not items: continue
        eta=estimate_eta(items)
        ranked=sorted(items, key=lambda x: float(x.get('sim_score') or 0.0), reverse=True)[:k]
        # Use realized utilities of the selected slate as a naive DM
        dm = float(np.mean([(ri.get('actual_score') or 0.0) - (ri.get('test_spend') or 0.0)/30.0 for ri in ranked])) if ranked else 0.0
        # IPS correction with exam prob
        pr=[(eta**i) for i in range(k)]
        ips = float(np.sum([pr[i]*((ri.get('actual_score') or 0.0) - (ri.get('test_spend') or 0.0)/30.0) for i,ri in enumerate(ranked)]))
        res.append({'campaign_file': f.name, 'eta': eta, 'dm_slate': dm, 'ips_exam': ips})
    OUT.joinpath('cascade_dr_topk.json').write_text(json.dumps({'results': res}, indent=2))
    print(json.dumps({'campaigns': len(res)}, indent=2))

if __name__=='__main__':
    main()
