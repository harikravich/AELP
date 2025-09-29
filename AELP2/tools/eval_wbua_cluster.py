#!/usr/bin/env python3
from __future__ import annotations
"""
Cluster-holdout WBUA: treat creative "families" as clusters using ad name
heuristics from creative_objects. Evaluate only items in week t whose family
did not appear in weeks t-4..t-1 for the same campaign.

Outputs: AELP2/reports/wbua_cluster_summary.json
"""
import json, math, re, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEEKLY = ROOT/ 'AELP2' / 'reports' / 'weekly_creatives'
COBJ   = ROOT/ 'AELP2' / 'reports' / 'creative_objects'
OUT = ROOT/ 'AELP2' / 'reports' / 'wbua_cluster_summary.json'

def family_of(creative_id: str) -> str|None:
    fp = COBJ / f"{creative_id}.json"
    if not fp.exists():
        return None
    try:
        d=json.loads(fp.read_text())
    except Exception:
        return None
    name=((d.get('ad') or {}).get('name')) or ((d.get('creative') or {}).get('name')) or ''
    # family heuristic: underscore-separated prefix of 2 tokens, or first 12 chars fallback
    if '_' in name:
        toks=name.split('_')
        fam='_'.join(toks[:2]).strip()
        return fam or name[:12]
    return name[:16] if name else None

def actual_cac(it):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def utility(it, target):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def boot_ci(vals, B=2000, seed=0):
    if not vals: return None, None
    random.seed(seed)
    stats=[]
    for _ in range(B):
        stats.append(random.choice(vals))
    stats.sort(); lo=stats[int(0.025*B)]; hi=stats[int(0.975*B)]
    return lo, hi

def main():
    files=sorted(WEEKLY.glob('*.json'))
    by_c={}
    for f in files:
        m=re.match(r'^(\d+)_([0-9]{4}W[0-9]{2})\.json$', f.name)
        if not m: continue
        cid=m.group(1)
        by_c.setdefault(cid, []).append(f)

    pairs=[]; group_acc=[]; counted=0
    for cid, flist in by_c.items():
        weeks=[json.loads(Path(ff).read_text()) for ff in sorted(flist)]
        for i in range(len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            target=sorted(cacs)[len(cacs)//2] if cacs else 200.0
            j0=max(0,i-4); j1=i-1
            if j1<j0: continue
            util_by_cre={}; last={}; prev_fams=set()
            for j in range(j0, j1+1):
                wk=weeks[j]
                for it in (wk.get('items') or []):
                    cr=it.get('creative_id'); prev_fams.add(family_of(cr))
                    u=utility(it, target)
                    util_by_cre[cr]=util_by_cre.get(cr,0.0)+u
                    last[cr]=it
            if not util_by_cre:
                continue
            base_id=max(util_by_cre.items(), key=lambda kv: kv[1])[0]
            base=last[base_id]
            b_sim=float(base.get('sim_score') or 0.0)
            b_p=float(base.get('actual_score') or 0.0)
            b_cac=actual_cac(base)
            corr=tot=0
            for v in items:
                fam=family_of(v.get('creative_id'))
                if fam in prev_fams:
                    continue  # hold out new families only
                sim_win=float(v.get('sim_score') or 0.0) > b_sim
                act_win=(float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)
                if sim_win==act_win: corr+=1
                tot+=1
                pairs.append(sim_win==act_win)
            if tot>0:
                group_acc.append(corr/tot)
                counted+=1

    wbua = (sum(1 for x in pairs if x)/len(pairs)) if pairs else None
    lo, hi = boot_ci(group_acc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'wbua_cluster_holdout': wbua, 'pairs': len(pairs), 'ci95_groups': [lo, hi], 'groups': len(group_acc), 'weeks_counted': counted}, indent=2))
    print(json.dumps({'wbua_cluster_holdout': wbua, 'pairs': len(pairs), 'groups': len(group_acc), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

