#!/usr/bin/env python3
from __future__ import annotations
"""
Export weekly Top-N creatives that pass the dual gate with a conservative lower-bound utility.

Inputs:
  - weekly_creatives/*.json
  - dual_gate_weekly.json (to pick operating point)
  - target_cac.json (optional; else median CAC per week)

Outputs:
  - AELP2/reports/weekly_topN_dual.json with entries: campaign_id, iso_week, items[{creative_id, adset_id, placement, lb_utility, pred_cac, pred_purchases}]
"""
import json, math, os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives'))
LOCK = ROOT / 'AELP2' / 'reports' / 'target_cac_locked.json'
DG = ROOT / 'AELP2' / 'reports' / 'dual_gate_weekly.json'
OUT = ROOT / 'AELP2' / 'reports' / 'weekly_topN_dual.json'

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def pred_cac(it: dict) -> float:
    s=float(it.get('test_spend') or 0.0); sim=float(it.get('sim_score') or 0.0)
    return (s/sim) if sim>0 else float('inf')

def median(xs: List[float]) -> float:
    if not xs: return float('nan')
    xs_sorted=sorted(xs)
    return xs_sorted[len(xs_sorted)//2]

def choose_operating_point(grid: Dict[str,dict], min_dual=0.6, min_yield=0.03) -> Dict[str, float]:
    # pick the first grid point with dual_precision >= min_dual and yield >= min_yield, prefer larger delta (stricter)
    # if none qualify, choose the point with highest dual_precision subject to minimal yield, else fallback d10_v10
    best_key=None; best_val=-1.0
    for k,v in grid.items():
        y=float(v.get('yield') or 0.0); dp=v.get('dual_precision');
        if dp is None: continue
        if dp>=min_dual and y>=min_yield:
            v['key']=k; return v
        if y>=min_yield and dp>best_val:
            best_val=dp; best_key=k
    if best_key:
        vv=grid[best_key]; vv['key']=best_key; return vv
    return {'key':'d10_v10','dual_precision': None}

def main():
    N=int(os.getenv('AELP2_DUAL_TOPN','10'))
    conf=json.loads(DG.read_text()) if DG.exists() else {'grid':{}}
    override=os.getenv('AELP2_DUAL_OP_KEY')
    if override and override in conf.get('grid',{}):
        op=conf['grid'][override]; op['key']=override
    else:
        op=choose_operating_point(conf.get('grid',{}))
    # Parse key like d10_v20
    delta=0.10; minv=20
    if op.get('key'):
        parts=op['key'].split('_');
        try:
            delta=float(parts[0][1:])/100.0; minv=int(parts[1][1:])
        except Exception:
            pass
    out={'topN': []}
    # optional locked targets
    locked = {}
    if LOCK.exists() and int(os.getenv('AELP2_USE_LOCKED_TARGETS','1'))==1:
        try:
            locked = (json.loads(LOCK.read_text()) or {}).get('targets') or {}
        except Exception:
            locked = {}
    # group files by campaign
    by_c: Dict[str, List[Path]]={}
    for f in sorted(WK.glob('*.json')):
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    for cid, flist in by_c.items():
        flist_sorted=sorted(flist, key=lambda p: p.name.split('_')[1])
        weeks=[json.loads(ff.read_text()) for ff in flist_sorted]
        for i in range(len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            if not items: continue
            if str(cid) in locked:
                target=float(locked[str(cid)])
            else:
                cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
                target=median(cacs) if cacs else 200.0
            thr=target*(1.0-delta)
            j0=max(0, i-4); j1=i-1
            if j1<j0: continue
            util_by_cre={}; last_entry={}
            for j in range(j0, j1+1):
                wk=weeks[j]
                for it in (wk.get('items') or []):
                    kcid=it.get('creative_id')
                    u=float(it.get('actual_score') or 0.0) - (float(it.get('test_spend') or 0.0)/target)
                    util_by_cre[kcid]=util_by_cre.get(kcid,0.0)+u
                    last_entry[kcid]=it
            if not util_by_cre: continue
            base_cre=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
            base=last_entry.get(base_cre)
            if not base: continue
            b_sim=float(base.get('sim_score') or 0.0)
            wk_label=flist_sorted[i].name.split('_')[1]
            sel=[]
            for it in items:
                if float(it.get('sim_score') or 0.0) <= b_sim: continue
                if pred_cac(it) > thr: continue
                if float(it.get('sim_score') or 0.0) < float(minv): continue
                lb_util = 0.8*float(it.get('sim_score') or 0.0) - (float(it.get('test_spend') or 0.0)/target)
                sel.append({
                    'creative_id': it.get('creative_id'),
                    'adset_id': it.get('adset_id'),
                    'placement': it.get('placement'),
                    'lb_utility': lb_util,
                    'pred_cac': pred_cac(it),
                    'pred_purchases': float(it.get('sim_score') or 0.0)
                })
            sel_sorted=sorted(sel, key=lambda x: x['lb_utility'], reverse=True)[:N]
            if sel_sorted:
                out['topN'].append({'campaign_id': cid, 'iso_week': wk_label, 'items': sel_sorted})
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'groups': len(out['topN']), 'avg_items': (sum(len(g['items']) for g in out['topN'])/max(1,len(out['topN']))) if out['topN'] else 0}, indent=2))

if __name__=='__main__':
    main()
