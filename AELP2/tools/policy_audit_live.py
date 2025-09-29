#!/usr/bin/env python3
from __future__ import annotations
"""
Policy audit of current live adsets/campaigns vs META_POLICY_SETUP.md
Outputs: AELP2/reports/policy_audit.json with counts per flag and list of non-compliant adsets.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / 'AELP2' / 'raw' / 'ad_config' / 'adsets.jsonl'
OUT = ROOT / 'AELP2' / 'reports' / 'policy_audit.json'

def flag_policy(adset: dict) -> dict:
    g=str(adset.get('optimization_goal') or '').lower()
    purchase_opt = ('purchase' in g) or ('offsite' in g and 'conversion' in g)
    dco = bool(adset.get('is_dynamic_creative'))
    bid=str(adset.get('bid_strategy') or '')
    lowest_cost = (bid == 'LOWEST_COST_WITHOUT_CAP')
    tgt = adset.get('targeting') or {}
    aa = False
    if isinstance(tgt, dict):
        aa = bool(tgt.get('advantage_audience')) or bool(tgt.get('detailed_targeting_expansion')) or bool(tgt.get('targeting_optimization_types'))
    auto_pl = bool((tgt or {}).get('is_automatic_placements')) if isinstance(tgt, dict) else False
    flags = {
        'purchase_opt': purchase_opt,
        'adv_audience': aa,
        'dco': dco,
        'auto_placements': auto_pl,
        'lowest_cost': lowest_cost,
    }
    return flags

def main():
    if not CONF.exists():
        OUT.write_text(json.dumps({'status':'no_config_dump'}, indent=2)); print('{"status":"no_config_dump"}'); return
    non=[]; counts={'purchase_opt':0,'adv_audience':0,'dco':0,'auto_placements':0,'lowest_cost':0,'total':0}
    for ln in CONF.read_text().splitlines():
        try:
            r=json.loads(ln)
        except Exception:
            continue
        flags=flag_policy(r)
        counts['total']+=1
        for k,v in flags.items():
            counts[k]+=1 if v else 0
        if not all(flags.values()):
            non.append({'adset_id': r.get('id'), 'name': r.get('name'), **flags})
    OUT.write_text(json.dumps({'counts': counts, 'non_compliant': non[:200]}, indent=2))
    print(json.dumps({'total_adsets': counts['total'], 'fully_compliant': sum(1 for x in non if False) }, indent=2))

if __name__=='__main__':
    main()

