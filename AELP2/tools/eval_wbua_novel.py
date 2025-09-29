#!/usr/bin/env python3
from __future__ import annotations
"""
Compute WBUA (Weekly Baseline Uplift Accuracy) for novel-only creatives.

Definition
  For each campaign-week t, pick the best historical baseline using weeks t-4..t-1
  (utility aggregation over those weeks). For week t items, mark a variant as a
  "win vs baseline" if it has >= actual_score and <= actual_cac relative to the
  baseline. Count whether the simulator's sim_score ranking agrees with that
  direction (sim_score higher than baseline => predict win). Only count items
  that did not appear in weeks t-4..t-1 for the same campaign (novel-only).

Outputs
  AELP2/reports/wbua_novel_summary.json
    { wbua_novel, pairs_novel, ci95, groups }
"""
import json, math, re, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEEKLY = ROOT/ 'AELP2' / 'reports' / 'weekly_creatives'
OUT = ROOT/ 'AELP2' / 'reports' / 'wbua_novel_summary.json'

def actual_cac(it):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def utility(it, target):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def bootstrap_ci(accs, B=2000, seed=0):
    random.seed(seed)
    stats=[]
    for _ in range(B):
        a=random.choice(accs)
        stats.append(a)
    stats.sort()
    lo=stats[int(0.025*B)]; hi=stats[int(0.975*B)]
    return lo, hi

def main():
    files=sorted(WEEKLY.glob('*.json'))
    # group by campaign id from filename prefix
    by_c={}
    for f in files:
        m=re.match(r'^(\d+)_([0-9]{4}W[0-9]{2})\.json$', f.name)
        if not m: continue
        cid=m.group(1)
        by_c.setdefault(cid, []).append(f)

    pairs_n=[]; group_acc=[]
    for cid, flist in by_c.items():
        weeks=[json.loads(Path(ff).read_text()) for ff in sorted(flist)]
        for i in range(len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            target=sorted(cacs)[len(cacs)//2] if cacs else 200.0
            j0=max(0,i-4); j1=i-1
            if j1<j0: continue
            util_by_cre={}; last={}; seen_prev=set()
            for j in range(j0, j1+1):
                wk=weeks[j]
                for it in (wk.get('items') or []):
                    cr=it.get('creative_id'); seen_prev.add(cr)
                    u=utility(it, target)
                    util_by_cre[cr]=util_by_cre.get(cr,0.0)+u
                    last[cr]=it
            if not util_by_cre: continue
            base_id=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
            base=last[base_id]
            b_sim=float(base.get('sim_score') or 0.0)
            b_p=float(base.get('actual_score') or 0.0)
            b_cac=actual_cac(base)
            corr=tot=0
            for v in items:
                if v.get('creative_id') in seen_prev:  # novel-only
                    continue
                sim_win=float(v.get('sim_score') or 0.0) > b_sim
                act_win=(float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)
                if sim_win==act_win: corr+=1
                tot+=1
                pairs_n.append(sim_win==act_win)
            if tot>0:
                group_acc.append(corr/tot)

    wbua_novel = (sum(1 for x in pairs_n if x)/len(pairs_n)) if pairs_n else None
    lo, hi = bootstrap_ci(group_acc) if group_acc else (None,None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'wbua_novel': wbua_novel,
        'pairs_novel': len(pairs_n),
        'ci95_groups': [lo, hi],
        'groups': len(group_acc)
    }, indent=2))
    print(json.dumps({'wbua_novel': wbua_novel, 'pairs_novel': len(pairs_n), 'groups': len(group_acc), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

