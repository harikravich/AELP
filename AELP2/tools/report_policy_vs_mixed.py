#!/usr/bin/env python3
from __future__ import annotations
"""
Compare WBUA and dual-gate metrics on mixed vs policy-annotated data.
Writes AELP2/reports/policy_vs_mixed_summary.json
"""
import json, os, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WK_MIX = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
WK_POL = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_policy'
OUT = ROOT / 'AELP2' / 'reports' / 'policy_vs_mixed_summary.json'

def actual_cac(it):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def utility(it, target):
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target if target>0 else 0.0)

def wbua(dirpath: Path, policy_only=False):
    files=sorted(dirpath.glob('*.json'))
    by_c={}
    for f in files:
        cid=f.name.split('_')[0]
        by_c.setdefault(cid, []).append(f)
    groups=[]
    for cid, fl in by_c.items():
        fl=sorted(fl, key=lambda p: p.name.split('_')[1])
        weeks=[json.loads(ff.read_text()) for ff in fl]
        for i in range(len(weeks)):
            cur=weeks[i]
            items=cur.get('items') or []
            if policy_only and items and 'policy_compliant' in items[0]:
                items=[it for it in items if it.get('policy_compliant')]
            if not items: continue
            cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            target=sorted(cacs)[len(cacs)//2] if cacs else 200.0
            j0=max(0, i-4); j1=i-1
            if j1<j0: continue
            util_by_cre={}; last_entry={}
            for j in range(j0, j1+1):
                wk=weeks[j]
                its=wk.get('items') or []
                if policy_only and its and 'policy_compliant' in its[0]:
                    its=[it for it in its if it.get('policy_compliant')]
                for it in its:
                    cid_it=it.get('creative_id');
                    util_by_cre[cid_it]=util_by_cre.get(cid_it,0.0)+utility(it,target)
                    last_entry[cid_it]=it
            if not util_by_cre: continue
            base=last_entry.get(max(util_by_cre.items(), key=lambda kv: kv[1])[0])
            if not base: continue
            b_sim=float(base.get('sim_score') or 0.0)
            b_p=float(base.get('actual_score') or 0.0)
            b_cac=actual_cac(base)
            corr=tot=0
            for v in items:
                sim_w = float(v.get('sim_score') or 0.0) > b_sim
                act_w = (float(v.get('actual_score') or 0.0) >= b_p) and (actual_cac(v) <= b_cac)
                if sim_w==act_w: corr+=1
                tot+=1
            if tot>0: groups.append((corr,tot))
    if not groups: return {'wbua': None, 'pairs': 0, 'ci95': [None,None], 'groups': 0}
    pairs=sum(t for _,t in groups)
    acc=sum(c for c,_ in groups)/pairs
    stats=[]; B=1000; random.seed(0)
    for _ in range(B):
        sample=[groups[random.randrange(len(groups))] for __ in range(len(groups))]
        s_pairs=sum(t for _,t in sample)
        s_acc=(sum(c for c,_ in sample)/s_pairs) if s_pairs>0 else 0.0
        stats.append(s_acc)
    lo=sorted(stats)[int(0.025*B)]; hi=sorted(stats)[int(0.975*B)]
    return {'wbua': acc, 'pairs': pairs, 'ci95': [lo,hi], 'groups': len(groups)}

def main():
    out={
        'mixed': wbua(WK_MIX, policy_only=False),
        'policy_only': wbua(WK_POL, policy_only=True)
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()

