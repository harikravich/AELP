#!/usr/bin/env python3
from __future__ import annotations
"""
Selection-conditional conformal bounds for weekly creatives.

Method (simple):
  - For each campaign, split its weekly files into calibration (all but last 2 weeks) and evaluation (last 2 weeks).
  - For calibration, compute residuals r = actual_purchases - sim_score for items that would be selected by a simple
    selection rule: top-K by sim_score (K from env, default 5).
  - Let q = quantile_{1-alpha} of |r| (alpha from env, default 0.1 → 90% lower bound).
  - For evaluation, report coverage that actual_purchases >= sim_score - q for the selected items.

Outputs: AELP2/reports/conformal_topk_weekly.json
"""
import json, math, os
from pathlib import Path
from typing import List, Dict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
OUT = ROOT / 'AELP2' / 'reports' / 'conformal_topk_weekly.json'

def quantile(arr: List[float], q: float) -> float:
    if not arr: return float('nan')
    a=sorted(arr)
    idx=int(q*(len(a)-1))
    return a[idx]

def main():
    K=int(os.getenv('AELP2_CONF_K','5'))
    alpha=float(os.getenv('AELP2_CONF_ALPHA','0.1'))
    files=sorted(WK.glob('*.json'))
    by_c={}
    for f in files:
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    cov_ct=0; total_ct=0; qvals=[]; per_c=[]
    for cid, flist in by_c.items():
        fl=sorted(flist, key=lambda p: p.name.split('_')[1])
        if len(fl)<4: continue
        cal=fl[:-2]; ev=fl[-2:]
        resid=[]
        for f in cal:
            d=json.loads(f.read_text()); items=d.get('items') or []
            sel=sorted(items, key=lambda x: float(x.get('sim_score') or 0.0), reverse=True)[:K]
            for it in sel:
                r = float(it.get('actual_score') or 0.0) - float(it.get('sim_score') or 0.0)
                resid.append(abs(r))
        q = quantile(resid, 1.0-alpha) if resid else float('nan')
        qvals.append(q if not math.isnan(q) else 0.0)
        # evaluation coverage
        for f in ev:
            d=json.loads(f.read_text()); items=d.get('items') or []
            sel=sorted(items, key=lambda x: float(x.get('sim_score') or 0.0), reverse=True)[:K]
            for it in sel:
                lb = float(it.get('sim_score') or 0.0) - (q if not math.isnan(q) else 0.0)
                ok = float(it.get('actual_score') or 0.0) >= lb
                cov_ct += 1 if ok else 0
                total_ct += 1
        per_c.append({'campaign_id': cid, 'q': q, 'calibration_n': len(resid)})
    cov = (cov_ct/total_ct) if total_ct>0 else None
    OUT.write_text(json.dumps({'K':K,'alpha':alpha,'coverage':cov,'n_eval':total_ct,'per_campaign':per_c}, indent=2))
    print(json.dumps({'coverage': cov, 'n_eval': total_ct}, indent=2))

if __name__=='__main__':
    main()

