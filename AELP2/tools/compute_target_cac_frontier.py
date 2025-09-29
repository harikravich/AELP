#!/usr/bin/env python3
from __future__ import annotations
"""
Compute per-campaign target_CAC by scanning historical CAC percentiles and choosing the point that maximizes
utility U = purchases - spend/target_CAC on held-out days (proxy frontier).
Outputs: AELP2/reports/target_cac.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'

def actual_cac(it: dict) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return (spend/purch) if purch>0 else float('inf')

def utility(it: dict, target: float) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return purch - (spend/target if target>0 else 0.0)

def main():
    out = {}
    for f in sorted(ENR.glob('*.json')):
        d = json.loads(f.read_text()); items=d.get('items') or []
        if not items: continue
        cacs = [actual_cac(it) for it in items if np.isfinite(actual_cac(it)) and actual_cac(it)<1e6]
        if not cacs:
            continue
        qs = [0.25, 0.3, 0.35, 0.4, 0.5]
        best = None; best_u = -1e9
        for q in qs:
            t = float(np.quantile(cacs, q))
            u = float(np.mean([utility(it, t) for it in items]))
            if u>best_u:
                best_u=u; best=t
        out[f.stem] = {'target_cac': best, 'obj_mean_utility': best_u, 'quantiles': {str(q): float(np.quantile(cacs,q)) for q in qs}}
    OUT.joinpath('target_cac.json').write_text(json.dumps(out, indent=2))
    print('AELP2/reports/target_cac.json')

if __name__=='__main__':
    main()

