#!/usr/bin/env python3
from __future__ import annotations
"""
OPE with config-based propensity reweighting (diagnostic).

Reads weekly_creatives_policy/*.json and computes a Top-K utility estimate where each item is reweighted by its
policy compliance propensity (approximate): p_conf = (policy_score + eps).

Outputs: AELP2/reports/ope_config_weighted.json
"""
import json, os, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives_policy'))
OUT = ROOT / 'AELP2' / 'reports' / 'ope_config_weighted.json'

def utility(it: dict, target_cac: float) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return p - (s/target_cac if target_cac>0 else 0.0)

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def main():
    k=int(os.getenv('AELP2_OPE_TOPK','5'))
    eps=1e-3
    res=[]
    for f in sorted(WK.glob('*.json')):
        d=json.loads(f.read_text())
        items=d.get('items') or []
        if not items: continue
        # target CAC: prefer policy median
        tgt=d.get('target_cac_policy')
        if tgt is None:
            cvals=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            tgt=float(sorted(cvals)[len(cvals)//2]) if cvals else 200.0
        else:
            tgt=float(tgt)
        ranked=sorted(items, key=lambda x: float(x.get('sim_score') or 0.0), reverse=True)[:k]
        # weights: inverse of (1 - policy_score) to upweight compliant; cap for stability
        ws=[]; vals=[]
        for it in ranked:
            sc=float(it.get('policy_score') or 0.0)
            w = (sc+eps)/(1.0-sc+eps)
            w = float(np.clip(w, 0.25, 4.0))
            ws.append(w)
            vals.append(utility(it, tgt))
        ws=np.array(ws); vals=np.array(vals)
        est=float(np.sum(ws*vals)/np.sum(ws)) if np.sum(ws)>0 else 0.0
        res.append({'campaign_file': f.name, 'k': k, 'target_cac': tgt, 'weighted_utility': est})
    OUT.write_text(json.dumps({'results': res}, indent=2))
    print(json.dumps({'campaigns': len(res)}, indent=2))

if __name__=='__main__':
    main()

