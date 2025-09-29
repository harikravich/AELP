#!/usr/bin/env python3
from __future__ import annotations
"""
OPE upgrades: DM, IPS, SNIPS, DR, SWITCH, CAB; plus simple bandit-FQE.

Inputs: creative_enriched/*.json with fields: sim_score, actual_score (purchases), test_spend, placement_stats.
We evaluate a Top-K policy that ranks by sim_score (or a provided score) and estimate expected utility
U = purchases - spend/target_CAC per campaign using multiple OPE estimators.

Outputs: AELP2/reports/ope_upgrades_topk.json
"""
import json, math, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import os
ENR = Path(os.getenv('AELP2_CREATIVE_DIR') or (ROOT / 'AELP2' / 'reports' / 'creative_enriched'))
OUT = ROOT / 'AELP2' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

def actual_cac(it: dict) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return (spend / purch) if purch > 0 else float('inf')

def utility(it: dict, target_cac: float) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return purch - (spend/target_cac if target_cac>0 else 0.0)

def topk_scores(items: List[dict], k: int, key='sim_score') -> Tuple[List[str], Dict[str,float]]:
    scored = sorted(items, key=lambda x: float(x.get(key) or 0.0), reverse=True)
    ids = [it.get('creative_id') for it in scored[:k]]
    score_map = {it.get('creative_id'): float(it.get(key) or 0.0) for it in items}
    return ids, score_map

def propensities_from_impressions(items: List[dict]) -> Dict[str, float]:
    imps = []
    for it in items:
        stats = it.get('placement_stats') or {}
        imps.append(sum((v or {}).get('impr',0.0) for v in stats.values()))
    s = sum(imps) or 1.0
    return {items[i].get('creative_id'): (imps[i]/s) for i in range(len(items))}

def estimators(values: Dict[str,float], policy: List[str], prop: Dict[str,float]) -> Dict[str,float]:
    # values: realized utility per ad id; policy: chosen ids; prop: logging propensities per id
    ids = list(values.keys())
    pset = set(policy)
    w = []
    v = []
    for i in ids:
        p = max(1e-8, float(prop.get(i, 0.0)))
        wi = (1.0/p) if (i in pset) else 0.0
        w.append(wi)
        v.append(values[i])
    w = np.array(w); v = np.array(v)
    # IPS/SNIPS
    ips = float(np.mean(w*v))
    snips = float(np.sum(w*v)/max(1e-8, np.sum(w)))
    # Direct method: constant predictor from logging (mean utility)
    dm = float(np.mean(v))
    # Doubly Robust: DM + weighted residuals
    dr = float(dm + np.mean(w*(v - dm)))
    # SWITCH: use IPS when w <= tau else DM; choose tau to minimize variance (simple sweep)
    taus = [1.0, 2.0, 5.0, 10.0]
    best = dr; best_tau = None
    for tau in taus:
        mask = (w <= tau).astype(float)
        sw = float(np.mean(mask*w*v + (1-mask)*dm))
        # choose tau by smallest empirical variance of the per-item terms
        terms = mask*w*v + (1-mask)*dm
        var = float(np.var(terms))
        if (best_tau is None) or (var < best):
            best = sw; best_tau = tau
    switch = best
    # CAB: continuous blending between IPS and DM by weight c in [0,1]
    # choose c to minimize variance of c*IPS + (1-c)*DM estimator terms
    cs = [0.25, 0.5, 0.75]
    cab = dr
    best_var = float('inf')
    ips_terms = w*v
    dm_terms = np.full_like(ips_terms, dm)
    for c in cs:
        terms = c*ips_terms + (1-c)*dm_terms
        var = float(np.var(terms))
        if var < best_var:
            best_var = var
            cab = float(np.mean(terms))
    return {'IPS': ips, 'SNIPS': snips, 'DM': dm, 'DR': dr, 'SWITCH': switch, 'CAB': cab}

def bandit_fqe(items: List[dict], policy: List[str], target_cac: float) -> float:
    # For a bandit, FQE reduces to a regression estimate of E[U|x,a]; here we approximate with a per-campaign mean utility
    # and evaluate on the chosen actions (policy). This is a simple DM.
    values = [utility(it, target_cac) for it in items if it.get('creative_id') in policy]
    return float(np.mean(values)) if values else 0.0

def main():
    k = int(os.getenv('AELP2_OPE_TOPK', '5'))
    results = []
    for f in sorted(ENR.glob('*.json')):
        d = json.loads(f.read_text())
        items = d.get('items') or []
        if not items:
            continue
        # target CAC = campaign median CAC as fallback
        cvals = []
        for it in items:
            c = actual_cac(it)
            if math.isfinite(c): cvals.append(c)
        target_cac = float(sorted(cvals)[len(cvals)//2]) if cvals else 30.0
        policy_ids, _ = topk_scores(items, k, key='sim_score')
        prop = propensities_from_impressions(items)
        val_map = {it.get('creative_id'): utility(it, target_cac) for it in items}
        est = estimators(val_map, policy_ids, prop)
        fqe = bandit_fqe(items, policy_ids, target_cac)
        results.append({'campaign_file': f.name, 'k': k, 'target_cac': target_cac, **est, 'FQE_DM': fqe})
    out = {'results': results}
    OUT.joinpath('ope_upgrades_topk.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({'campaigns': len(results)}, indent=2))

if __name__ == '__main__':
    main()
