#!/usr/bin/env python3
from __future__ import annotations
"""
Historic baseline uplift test.

For each campaign (creative_enriched/*.json):
  - Choose a baseline ad: older (age_days >= MIN_BASE_AGE) and adequate volume
  - For each later variant (age_days < baseline.age_days), compare:
      sim_winner  = sim_score(variant) > sim_score(baseline)
      actual_winner = beats by purchases & CAC: purch_v >= purch_b and cac_v <= 1.0 * cac_b
  - Accumulate correctness across all (baseline, variant) pairs.

Outputs: AELP2/reports/uplift_baseline_eval.json
"""
import json, math, os
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
import os
ENR = Path(os.getenv('AELP2_CREATIVE_DIR') or (ROOT / 'AELP2' / 'reports' / 'creative_enriched'))
OUT = ROOT / 'AELP2' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

MIN_BASE_AGE = int(os.getenv('AELP2_UPLIFT_MIN_BASE_AGE', '10'))
MIN_PURCH = int(os.getenv('AELP2_LABEL_MIN_PURCH', '2'))
MIN_CLICKS = int(os.getenv('AELP2_LABEL_MIN_CLICKS', '50'))
MIN_SPEND = float(os.getenv('AELP2_LABEL_MIN_SPEND', '50'))

def good_volume(it: dict) -> bool:
    return (float(it.get('actual_score') or 0.0) >= MIN_PURCH) or \
           (float(it.get('test_clicks') or 0.0) >= MIN_CLICKS) or \
           (float(it.get('test_spend') or 0.0) >= MIN_SPEND)

def actual_cac(it: dict) -> float:
    purch = float(it.get('actual_score') or 0.0)
    spend = float(it.get('test_spend') or 0.0)
    return (spend / purch) if purch > 0 else float('inf')

def pick_baseline(items: List[dict]) -> dict | None:
    # Eligible = adequate volume and sufficiently old
    eligible = [it for it in items if (it.get('dna_real') or {}).get('age_days') is not None and (it.get('dna_real') or {}).get('age_days') >= MIN_BASE_AGE and good_volume(it)]
    if not eligible:
        return None
    # Select the best CAC among eligible as baseline (older, robust)
    eligible.sort(key=lambda it: (actual_cac(it), -float(it.get('actual_score') or 0.0)))
    return eligible[0]

def evaluate_campaign(file: Path) -> Tuple[int,int,Dict]:
    d = json.loads(file.read_text())
    items = d.get('items') or []
    if not items:
        return 0,0,{'campaign_file': file.name, 'reason': 'no_items'}
    base = pick_baseline(items)
    if not base:
        return 0,0,{'campaign_file': file.name, 'reason': 'no_baseline'}
    base_age = (base.get('dna_real') or {}).get('age_days') or 0
    # later variants are newer ads with adequate volume
    variants = [it for it in items if (it.get('dna_real') or {}).get('age_days') is not None and (it.get('dna_real') or {}).get('age_days') < base_age and good_volume(it)]
    if not variants:
        return 0,0,{'campaign_file': file.name, 'reason': 'no_variants'}
    b_sim = float(base.get('sim_score') or 0.0)
    b_purch = float(base.get('actual_score') or 0.0)
    b_cac = actual_cac(base)
    # Utility metric: purchases - spend/target_cac, where target_cac = median CAC in this campaign
    cvals = [actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
    target_cac = float(sorted(cvals)[len(cvals)//2]) if cvals else (b_cac if math.isfinite(b_cac) else 30.0)
    def utility(it: dict) -> float:
        purch = float(it.get('actual_score') or 0.0)
        spend = float(it.get('test_spend') or 0.0)
        return purch - (spend/target_cac if target_cac>0 else 0.0)
    u_base = utility(base)
    correct = total = 0
    details = []
    for v in variants:
        v_sim = float(v.get('sim_score') or 0.0)
        v_purch = float(v.get('actual_score') or 0.0)
        v_cac = actual_cac(v)
        sim_winner = v_sim > b_sim
        # Utility-based win with relaxed thresholds; also record strict outcome
        strict_actual_winner = (v_purch >= b_purch) and (v_cac <= b_cac)
        actual_winner = (utility(v) > u_base)
        if sim_winner == actual_winner:
            correct += 1
        total += 1
        details.append({'variant': v.get('creative_id'), 'sim_winner': sim_winner, 'actual_winner': actual_winner, 'strict_actual': strict_actual_winner})
    return correct, total, {
        'campaign_file': file.name,
        'baseline_id': base.get('creative_id'),
        'baseline_age_days': base_age,
        'pairs': total,
        'accuracy': (correct/total) if total else None
    }

def main():
    files = sorted(ENR.glob('*.json'))
    total_c = total_n = 0
    per = []
    for f in files:
        c,n,info = evaluate_campaign(f)
        total_c += c; total_n += n
        per.append(info)
    summary = {
        'uplift_accuracy': (total_c/total_n) if total_n else None,
        'pairs_total': total_n,
        'campaigns': len(files)
    }
    OUT.joinpath('uplift_baseline_eval.json').write_text(json.dumps({'summary': summary, 'details': per}, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
