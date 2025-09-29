#!/usr/bin/env python3
from __future__ import annotations
"""
Join adset/campaign config into weekly creatives and mark policy compliance.

Inputs:
  - AELP2/raw/ad_config/adsets.jsonl (from export_meta_ad_configs.py)
  - AELP2/reports/weekly_creatives/*.json

Outputs:
  - AELP2/reports/weekly_creatives_policy/*.json
    Each item gains: policy_flags{purchase_opt,adv_audience,dco,auto_placements,lowest_cost}, policy_compliant(bool), compliance_score(0..1)
    Top-level adds: target_cac_policy (median CAC among compliant items), policy_summary counts.
"""
import json, math, os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'AELP2' / 'raw' / 'ad_config'
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
OUT = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_policy'
OUT.mkdir(parents=True, exist_ok=True)

def load_adsets():
    m={}
    fp=RAW/'adsets.jsonl'
    if not fp.exists():
        return m
    for ln in fp.read_text().splitlines():
        try:
            r=json.loads(ln)
        except Exception:
            continue
        m[str(r.get('id'))]=r
    return m

def flag_policy(adset: dict) -> dict:
    g=str(adset.get('optimization_goal') or '').lower()
    purchase_opt = ('purchase' in g) or ('offsite' in g and 'conversion' in g)
    dco = bool(adset.get('is_dynamic_creative'))
    bid=str(adset.get('bid_strategy') or '')
    lowest_cost = (bid == 'LOWEST_COST_WITHOUT_CAP')
    tgt = adset.get('targeting') or {}
    # Advantage Audience signals (best-effort)
    aa = False
    if isinstance(tgt, dict):
        aa = bool(tgt.get('advantage_audience')) or bool(tgt.get('detailed_targeting_expansion'))
        # legacy field names observed in some exports
        aa = aa or bool(tgt.get('targeting_optimization_types'))
    # Advantage+ placements (auto placements)
    auto_pl = bool(tgt.get('is_automatic_placements')) if isinstance(tgt, dict) else False
    # If field missing, treat many placements as likely auto
    if not auto_pl and isinstance(tgt, dict):
        plats = tgt.get('publisher_platforms') or []
        auto_pl = bool(plats) and len(plats) >= 3
    flags = {
        'purchase_opt': purchase_opt,
        'adv_audience': aa,
        'dco': dco,
        'auto_placements': auto_pl,
        'lowest_cost': lowest_cost,
    }
    score = sum(1.0 if v else 0.0 for v in flags.values())/len(flags)
    return {'flags': flags, 'score': score, 'compliant': all(flags.values())}

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def main():
    adsets=load_adsets()
    files=sorted(WK.glob('*.json'))
    for f in files:
        d=json.loads(f.read_text())
        items=d.get('items') or []
        comp=[]; annotated=[]; n_c=0
        for it in items:
            aid=str(it.get('adset_id') or '')
            conf=adsets.get(aid) or {}
            pol=flag_policy(conf) if conf else {'flags':{}, 'score':0.0, 'compliant': False}
            x=dict(it)
            x['policy_flags']=pol['flags']
            x['policy_compliant']=pol['compliant']
            x['policy_score']=pol['score']
            annotated.append(x)
            if pol['compliant']:
                c=actual_cac(x)
                if math.isfinite(c):
                    comp.append(c)
                n_c+=1
        tgt = float(median(comp)) if comp else None
        out={'campaign_id': d.get('campaign_id'), 'iso_week': d.get('iso_week'), 'items': annotated,
             'target_cac_policy': tgt,
             'policy_summary': {'compliant_items': n_c, 'total_items': len(items)}}
        (OUT/f.name).write_text(json.dumps(out, indent=2))
        print(f"wrote {OUT/f.name} (items={len(items)}, compliant={n_c}, tgt={tgt})")

if __name__=='__main__':
    main()

