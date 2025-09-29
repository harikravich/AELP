#!/usr/bin/env python3
from __future__ import annotations
"""
Dual-gate weekly evaluator.

Gate a variant in week t if ALL hold:
  1) predicted CAC <= target_CAC * (1 - delta)
  2) predicted purchases >= min_volume
  3) sim_score_variant > sim_score_baseline_slate (baseline chosen from previous up-to-4 weeks by max utility)

Metrics (aggregated over campaign-weeks):
  - precision_beat_baseline: fraction of gated variants whose actual beats baseline (p>=base_p and CAC<=base_CAC)
  - precision_meet_target: fraction of gated variants whose actual CAC <= target*(1-delta)
  - dual_precision: fraction meeting BOTH above
  - yield: gated variants / total variant comparisons
  - 95% bootstrap CIs for each precision

Outputs: AELP2/reports/dual_gate_weekly.json
"""
import json, math, os, random
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives'))
LOCK = ROOT / 'AELP2' / 'reports' / 'target_cac_locked.json'
OUT = ROOT / 'AELP2' / 'reports' / 'dual_gate_weekly.json'
CAL = ROOT / 'AELP2' / 'reports' / 'cac_calibrator.json'

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def pred_cac(it: dict) -> float:
    # Prefer CPC-mix-based predicted CAC if provided, else spend/sim proxy
    use_cpc = int(os.getenv('AELP2_USE_CPC','1'))==1
    if use_cpc and (it.get('pred_cac_cpc') is not None):
        try:
            return float(it.get('pred_cac_cpc'))
        except Exception:
            pass
    s=float(it.get('test_spend') or 0.0); sim=float(it.get('sim_score') or 0.0)
    return (s/sim) if sim>0 else float('inf')

def load_calibrator():
    if CAL.exists() and int(os.getenv('AELP2_USE_CAL','1'))==1:
        try:
            d=json.loads(CAL.read_text())
            xs=d.get('x'); ys=d.get('y')
            if xs and ys and len(xs)==len(ys):
                return list(map(float,xs)), list(map(float,ys))
        except Exception:
            return None
    return None

def apply_calibrator(x: float, cal):
    if not cal: return x
    xs, ys = cal
    if not xs: return x
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    import bisect
    i=bisect.bisect_left(xs, x)
    x0,x1=xs[i-1],xs[i]; y0,y1=ys[i-1],ys[i]
    t=(x-x0)/(x1-x0) if x1> x0 else 0.0
    return y0 + t*(y1-y0)

def utility(it: dict, target: float) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def median(xs: List[float]) -> float:
    if not xs: return float('nan')
    xs_sorted=sorted(xs)
    return xs_sorted[len(xs_sorted)//2]

def load_week_files() -> Dict[str, List[Path]]:
    by_c={}
    for f in sorted(WK.glob('*.json')):
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    for k in by_c:
        by_c[k]=sorted(by_c[k], key=lambda p: p.name.split('_')[1])
    return by_c

def evaluate(delta_vals=(0.05,0.10,0.15,0.20), min_vol_vals=(10,20,30)):
    by_c = load_week_files()
    locked = {}
    if LOCK.exists() and int(os.getenv('AELP2_USE_LOCKED_TARGETS','1'))==1:
        try:
            locked = (json.loads(LOCK.read_text()) or {}).get('targets') or {}
        except Exception:
            locked = {}
    grid_results = {}
    cal = load_calibrator()
    for delta in delta_vals:
        for min_v in min_vol_vals:
            key=f"d{int(delta*100)}_v{min_v}"
            # collect per-group tallies (tp_dual, tp_beat, tp_target, selected, total)
            group_stats=[]
            for cid, flist in by_c.items():
                weeks=[json.loads(ff.read_text()) for ff in flist]
                for i in range(len(weeks)):
                    cur=weeks[i]
                    items=cur.get('items') or []
                    if int(os.getenv('AELP2_POLICY_ONLY','0'))==1 and items and 'policy_compliant' in items[0]:
                        items=[it for it in items if it.get('policy_compliant')]
                    if not items: continue
                    # current week target CAC (prefer policy median)
                    tgt_pol = cur.get('target_cac_policy')
                    if str(cur.get('campaign_id')) in locked:
                        target=float(locked[str(cur.get('campaign_id'))])
                    elif tgt_pol is not None:
                        target=float(tgt_pol)
                    else:
                        cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
                        target=median(cacs) if cacs else 200.0
                    # baseline slate from previous up-to-4 weeks
                    j0=max(0, i-4); j1=i-1
                    if j1<j0: 
                        continue
                    util_by_cre={}
                    last_entry={}
                    for j in range(j0, j1+1):
                        wk=weeks[j]
                        its = wk.get('items') or []
                        if int(os.getenv('AELP2_POLICY_ONLY','0'))==1 and its and 'policy_compliant' in its[0]:
                            its=[it for it in its if it.get('policy_compliant')]
                        for it in its:
                            kcid=it.get('creative_id')
                            util_by_cre[kcid]=util_by_cre.get(kcid,0.0)+utility(it, target)
                            last_entry[kcid]=it
                    if not util_by_cre:
                        continue
                    base_cre=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
                    base=last_entry.get(base_cre)
                    if not base:
                        continue
                    b_sim=float(base.get('sim_score') or 0.0)
                    b_p=float(base.get('actual_score') or 0.0)
                    b_cac=actual_cac(base)
                    # gate variants
                    sel=0; tot=0; tp_dual=0; tp_beat=0; tp_target=0
                    thr = target*(1.0-delta)
                    for v in items:
                        tot+=1
                        if float(v.get('sim_score') or 0.0) <= b_sim:
                            continue
                        pc = pred_cac(v)
                        pc = apply_calibrator(pc, cal)
                        if pc > thr:
                            continue
                        if float(v.get('sim_score') or 0.0) < float(min_v):
                            # optional volume gate by predicted purchases (absolute)
                            continue
                        sel+=1
                        beat = (float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)
                        meet = (actual_cac(v) <= thr)
                        if beat: tp_beat+=1
                        if meet: tp_target+=1
                        if beat and meet: tp_dual+=1
                    if tot>0:
                        group_stats.append((tp_dual,tp_beat,tp_target,sel,tot))
            # aggregate and bootstrap CIs
            if not group_stats:
                grid_results[key]={'groups':0}
                continue
            agg = lambda idx: sum(g[idx] for g in group_stats)
            S=agg(3); T=agg(4)
            def ratio(num,den):
                return (num/den) if den>0 else None
            dual = ratio(agg(0), S)
            beat = ratio(agg(1), S)
            meet = ratio(agg(2), S)
            yield_ = ratio(S, T)
            # bootstrap
            B=1000; random.seed(0)
            dual_ci=[]; beat_ci=[]; meet_ci=[]
            for _ in range(B):
                sample=[group_stats[random.randrange(len(group_stats))] for __ in range(len(group_stats))]
                S_s=sum(s for _,_,_,s,_ in sample); T_s=sum(t for *_,t in sample)
                dual_s=(sum(x for x,_,_,_,_ in sample)/S_s) if S_s>0 else 0.0
                beat_s=(sum(x for _,x,_,_,_ in sample)/S_s) if S_s>0 else 0.0
                meet_s=(sum(x for _,_,x,_,_ in sample)/S_s) if S_s>0 else 0.0
                dual_ci.append(dual_s); beat_ci.append(beat_s); meet_ci.append(meet_s)
            ci = lambda arr: [sorted(arr)[int(0.025*B)], sorted(arr)[int(0.975*B)]]
            grid_results[key]={
                'groups': len(group_stats), 'selected': S, 'total': T,
                'dual_precision': dual, 'dual_ci95': ci(dual_ci),
                'precision_beat_baseline': beat, 'beat_ci95': ci(beat_ci),
                'precision_meet_target': meet, 'meet_ci95': ci(meet_ci),
                'yield': yield_
            }
    OUT.write_text(json.dumps({'grid': grid_results}, indent=2))
    print(json.dumps({'grid_keys': list(grid_results.keys())}, indent=2))

def main():
    deltas = tuple(float(x) for x in (os.getenv('AELP2_DUAL_DELTAS','0.05,0.10,0.15,0.20').split(',')))
    mins = tuple(int(x) for x in (os.getenv('AELP2_DUAL_MINV','10,20,30').split(',')))
    evaluate(deltas, mins)

if __name__=='__main__':
    main()
