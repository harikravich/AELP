#!/usr/bin/env python3
from __future__ import annotations
"""
Estimate coarse uplift scalars for key policy switches by comparing compliant-like vs noncompliant items within campaign-weeks.
Outputs: AELP2/reports/policy_uplift.json with ratios for CVR (purchases/clicks) and CAC.
"""
import json, math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_policy'
OUT = ROOT / 'AELP2' / 'reports' / 'policy_uplift.json'

def safe_div(a,b):
    return (a/b) if b>0 else float('nan')

def main():
    files=sorted(WK.glob('*.json'))
    ratios={'cvrs':[], 'cacs':[]}
    for f in files:
        d=json.loads(f.read_text()); items=d.get('items') or []
        if not items: continue
        comp=[it for it in items if (it.get('policy_score') or 0.0) >= 0.6]
        nonc=[it for it in items if (it.get('policy_score') or 0.0) < 0.6]
        if len(comp)<3 or len(nonc)<3: continue
        # campaign-week medians
        def stats(lst):
            clicks=[float(x.get('test_clicks') or 0.0) for x in lst]
            purch=[float(x.get('actual_score') or 0.0) for x in lst]
            spend=[float(x.get('test_spend') or 0.0) for x in lst]
            cvr=median([safe_div(pu,cl) for pu,cl in zip(purch,clicks) if cl>0] or [float('nan')])
            cac=median([safe_div(sp,pu) for pu,sp in zip(purch,spend) if pu>0] or [float('nan')])
            return cvr,cac
        c_cvr,c_cac = stats(comp)
        n_cvr,n_cac = stats(nonc)
        if not math.isnan(c_cvr) and not math.isnan(n_cvr) and n_cvr>0:
            ratios['cvrs'].append(c_cvr/n_cvr)
        if not math.isnan(c_cac) and not math.isnan(n_cac) and n_cac>0:
            ratios['cacs'].append(n_cac/c_cac)
    def summarize(arr):
        if not arr: return {'n':0}
        arr_sorted=sorted(arr)
        import numpy as np
        return {'n': len(arr), 'p50': float(np.percentile(arr_sorted,50)), 'p25': float(np.percentile(arr_sorted,25)), 'p75': float(np.percentile(arr_sorted,75))}
    out={'cvr_uplift_ratio': summarize(ratios['cvrs']), 'cac_gain_ratio': summarize(ratios['cacs'])}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()

