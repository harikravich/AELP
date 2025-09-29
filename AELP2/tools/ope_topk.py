#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CRE_DIR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'

def load_campaigns():
    files = sorted(CRE_DIR.glob('*.json'))
    camps=[]
    for f in files:
        d=json.loads(f.read_text())
        camps.append((f.name, d.get('items') or []))
    return camps

def estimate_snips_topk(items, scores, k=5):
    # Approximate propensities by impression share (from placement_stats if available)
    exposures = []
    rewards = []
    ids = []
    for it, s in zip(items, scores):
        stats = it.get('placement_stats') or {}
        impr = sum((v or {}).get('impr',0.0) for v in stats.values())
        exposures.append(impr)
        rewards.append(float(it.get('actual_score') or 0.0))
        ids.append(it.get('creative_id'))
    exp_sum = sum(exposures) or 1.0
    prop = [e/exp_sum for e in exposures]
    # Policy: choose top-K by scores
    order = np.argsort(-np.array(scores))
    chosen = set(ids[i] for i in order[:min(k, len(order))])
    # SNIPS: sum 1{a in policy} * r(a) / p(a) normalized
    numer = 0.0; denom = 0.0
    for i,aid in enumerate(ids):
        if prop[i] <= 0: continue
        w = 1.0/prop[i] if aid in chosen else 0.0
        numer += w*rewards[i]
        denom += w
    snips = numer/denom if denom>0 else 0.0
    return snips

def main():
    # Build simple score = sim_score baseline
    camps = load_campaigns()
    results=[]
    for name, items in camps:
        scores = [float(it.get('sim_score') or 0.0) for it in items]
        sn = estimate_snips_topk(items, scores, k=5)
        results.append({'campaign': name, 'snips_top5': sn, 'n': len(items)})
    OUT.joinpath('ope_topk_baseline.json').write_text(json.dumps({'results': results}, indent=2))
    print(json.dumps({'summary': {'campaigns': len(results)}}, indent=2))

if __name__=='__main__':
    main()

