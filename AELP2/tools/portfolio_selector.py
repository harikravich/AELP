#!/usr/bin/env python3
from __future__ import annotations
"""
Portfolio selector: choose a per-campaign slate that maximizes predicted purchases under CAC caps and test budget share.

Inputs
- weekly_creatives_cpc/*.json (preferred) or weekly_creatives/*.json
- target_cac_locked.json (if present)
- cac_calibrator.json (optional, applied if present)

Selection rule (per campaign-week)
- Operating point: read AELP2_DUAL_OP_KEY (e.g., d10_v30 → delta=0.10, min_volume=30)
- Gate: sim_score > baseline_sim, predicted CAC (with calibrator) ≤ target*(1-delta), sim_score ≥ min_volume
- Score: lower-bound utility = 0.8*sim_score - spend/target
- Pick top N (AELP2_PORTFOLIO_TOPN, default 10)

Outputs
- AELP2/reports/weekly_portfolio.json: [{campaign_id, iso_week, target_cac, delta, min_volume, items:[...], policy_playbook:"AELP2/docs/META_POLICY_SETUP.md"}]
"""
import json, math, os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
WK_CPC = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_cpc'
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
LOCK = ROOT / 'AELP2' / 'reports' / 'target_cac_locked.json'
CAL = ROOT / 'AELP2' / 'reports' / 'cac_calibrator.json'
OUT = ROOT / 'AELP2' / 'reports' / 'weekly_portfolio.json'

def load_cal():
    if CAL.exists():
        try:
            d=json.loads(CAL.read_text()); xs=d.get('x'); ys=d.get('y')
            if xs and ys and len(xs)==len(ys):
                return list(map(float,xs)), list(map(float,ys))
        except Exception: pass
    return None

def interp(x, cal):
    if not cal: return x
    xs,ys=cal
    if x<=xs[0]: return ys[0]
    if x>=xs[-1]: return ys[-1]
    import bisect
    i=bisect.bisect_left(xs, x)
    x0,x1=xs[i-1],xs[i]; y0,y1=ys[i-1],ys[i]
    t=(x-x0)/(x1-x0) if x1>x0 else 0.0
    return y0+t*(y1-y0)

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def median(xs: List[float]) -> float:
    if not xs: return float('nan')
    s=sorted(xs); return s[len(s)//2]

def main():
    topn=int(os.getenv('AELP2_PORTFOLIO_TOPN','10'))
    op=os.getenv('AELP2_DUAL_OP_KEY','d10_v30')
    parts=op.split('_'); delta=float(parts[0][1:])/100.0; minv=int(parts[1][1:])
    locked=(json.loads(LOCK.read_text()).get('targets') if LOCK.exists() else {}) or {}
    wk = WK_CPC if WK_CPC.exists() else WK
    files=sorted(wk.glob('*.json'))
    by_c={}
    for f in files:
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    cal=load_cal()
    out={'portfolio': []}
    for cid, fl in by_c.items():
        fl=sorted(fl, key=lambda p: p.name.split('_')[1])
        weeks=[json.loads(ff.read_text()) for ff in fl]
        for i in range(len(weeks)):
            cur=weeks[i]; items=cur.get('items') or []
            if not items: continue
            if str(cid) in locked:
                target=float(locked[str(cid)])
            else:
                cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
                target=median(cacs) if cacs else 200.0
            # baseline sim from previous up-to-4 weeks
            j0=max(0,i-4); j1=i-1
            b_sim=0.0
            if j1>=j0:
                util={}; last={}
                for j in range(j0,j1+1):
                    wkprev=weeks[j]; its=wkprev.get('items') or []
                    for it in its:
                        u=float(it.get('actual_score') or 0.0) - (float(it.get('test_spend') or 0.0)/target)
                        cid_it=it.get('creative_id'); util[cid_it]=util.get(cid_it,0.0)+u; last[cid_it]=it
                if util:
                    base=last[max(util.items(), key=lambda kv: kv[1])[0]]
                    b_sim=float(base.get('sim_score') or 0.0)
            thr=target*(1.0-delta)
            sel=[]
            for it in items:
                sim=float(it.get('sim_score') or 0.0)
                if sim<=b_sim or sim<minv: continue
                pc = it.get('pred_cac_cpc')
                if pc is None:
                    s=float(it.get('test_spend') or 0.0); simd=float(it.get('sim_score') or 0.0); pc=(s/simd) if simd>0 else float('inf')
                pc = float(pc)
                pc = interp(pc, cal)
                if pc>thr: continue
                lb = 0.8*sim - (float(it.get('test_spend') or 0.0)/target)
                sel.append({'creative_id': it.get('creative_id'),
                            'adset_id': it.get('adset_id'),
                            'placement': it.get('placement') or 'unknown',
                            'pred_cac': pc,
                            'pred_purchases': sim,
                            'lb_utility': lb})
            sel_sorted=sorted(sel, key=lambda x: x['lb_utility'], reverse=True)[:topn]
            if sel_sorted:
                out['portfolio'].append({'campaign_id': cid,
                                         'iso_week': cur.get('iso_week'),
                                         'target_cac': target,
                                         'delta': delta,
                                         'min_volume': minv,
                                         'policy_playbook': 'AELP2/docs/META_POLICY_SETUP.md',
                                         'items': sel_sorted})
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'groups': len(out['portfolio'])}, indent=2))

if __name__=='__main__':
    main()

