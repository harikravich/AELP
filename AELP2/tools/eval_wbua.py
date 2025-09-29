#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate Weekly Baseline Uplift Accuracy (WBUA) on weekly_creatives/* files.
For each campaign-week, choose baseline = best ad by previous week actual utility (purchases - spend/target_CAC),
then measure fraction of variants where (sim_score_variant > sim_score_baseline) matches (actual_variant beats baseline at equal/lower CAC).
Outputs: AELP2/reports/wbua_summary.json
"""
import json, math, os
from pathlib import Path
import os
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives'))
OUT = ROOT / 'AELP2' / 'reports' / 'wbua_summary.json'

def utility(it: dict, target: float) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def main():
    files=sorted(WK.glob('*.json'))
    # group by campaign
    by_c={}
    for f in files:
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    # per-group tallies: list of (correct, total) per campaign-week
    group_stats=[]
    for cid, flist in by_c.items():
        flist_sorted=sorted(flist, key=lambda p: p.name.split('_')[1])
        # Preload all weeks content
        weeks=[json.loads(ff.read_text()) for ff in flist_sorted]
        for i in range(len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            if int(os.getenv('AELP2_POLICY_ONLY','0'))==1 and items and 'policy_compliant' in items[0]:
                items=[it for it in items if it.get('policy_compliant')]
            if not items: continue
            # target CAC for current week: prefer policy median if present, else overall
            tgt_pol = cur.get('target_cac_policy')
            if tgt_pol is not None:
                target=float(tgt_pol)
            else:
                cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
                target=sorted(cacs)[len(cacs)//2] if cacs else 200.0
            # build baseline slate from previous up-to-4 weeks
            j0=max(0, i-4); j1=i-1
            if j1<j0:
                continue
            util_by_cre={}
            last_entry={}  # creative_id -> most recent item dict among baseline window
            for j in range(j0, j1+1):
                wk=weeks[j]
                its = wk.get('items') or []
                if int(os.getenv('AELP2_POLICY_ONLY','0'))==1 and its and 'policy_compliant' in its[0]:
                    its=[it for it in its if it.get('policy_compliant')]
                for it in its:
                    cid_it=it.get('creative_id')
                    u=utility(it, target)
                    util_by_cre[cid_it]=util_by_cre.get(cid_it, 0.0)+u
                    # track most recent occurrence for b_sim/b_p/b_cac reference
                    last_entry[cid_it]=it
            if not util_by_cre:
                continue
            base_cre=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
            base=last_entry.get(base_cre)
            if not base:
                continue
            b_sim= float(base.get('sim_score') or 0.0)
            b_p = float(base.get('actual_score') or 0.0)
            b_cac= actual_cac(base)
            g_correct=g_total=0
            for v in items:
                sim_winner = float(v.get('sim_score') or 0.0) > b_sim
                act_winner = (float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)
                if sim_winner == act_winner:
                    g_correct+=1
                g_total+=1
            if g_total>0:
                group_stats.append((g_correct, g_total))
    total_pairs=sum(t for _,t in group_stats)
    acc = (sum(c for c,_ in group_stats)/total_pairs) if total_pairs>0 else None
    # bootstrap 95% CI over groups
    import random
    random.seed(0)
    B=1000
    stats=[]
    if group_stats:
        for _ in range(B):
            sample=[group_stats[random.randrange(len(group_stats))] for __ in range(len(group_stats))]
            s_pairs=sum(t for _,t in sample)
            s_acc=(sum(c for c,_ in sample)/s_pairs) if s_pairs>0 else 0.0
            stats.append(s_acc)
        lo=sorted(stats)[int(0.025*B)]
        hi=sorted(stats)[int(0.975*B)]
    else:
        lo=hi=None
    OUT.write_text(json.dumps({'wbua': acc, 'pairs': total_pairs, 'ci95': [lo,hi], 'groups': len(group_stats)}, indent=2))
    print(json.dumps({'wbua': acc, 'pairs': total_pairs, 'ci95': [lo,hi], 'groups': len(group_stats)}))

if __name__=='__main__':
    main()
