#!/usr/bin/env python3
"""
Compute ad-level offline accuracy metrics from historical data and simulator predictions.

Inputs (best-effort; all local/offline):
  - AELP2/reports/sim_fidelity_campaigns.json (or *_temporal_v2.json)
  - AELP2/reports/creative/*.json (realized outcomes per creative_id)
  - AELP2/reports/rl_shadow_score.json (example simulated predictions per campaign)

Outputs:
  - AELP2/reports/ad_level_accuracy.json (Precision@K, pairwise win-rate, Kendall tau, Brier/AUC)
  - AELP2/reports/ad_level_accuracy.csv (per-campaign/per-date diagnostics)

This script is resilient: if some inputs are missing, it will compute what it can and annotate reasons.
"""
from __future__ import annotations
import json, os, math, csv
from pathlib import Path
from typing import Dict, List, Tuple

from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'AELP2' / 'reports'

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def kendall_tau(rank_a: List[str], rank_b: List[str]) -> float:
    # Simple Kendall tau on intersection
    common = [x for x in rank_a if x in set(rank_b)]
    n = len(common)
    if n < 2:
        return float('nan')
    index_b = {k:i for i,k in enumerate(rank_b)}
    concordant = discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            ai, aj = common[i], common[j]
            if index_b[ai] < index_b[aj]:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else float('nan')

def precision_at_k(sorted_ids: List[str], labels: Dict[str, int], k: int=5) -> float:
    topk = sorted_ids[:k]
    if not topk:
        return float('nan')
    return sum(labels.get(cid,0) for cid in topk) / len(topk)

def pairwise_win_rate(sorted_ids: List[str], labels: Dict[str, int]) -> float:
    # labels: 1 if "worked" live (e.g., beat control by CAC or conversions), else 0
    ids = [i for i in sorted_ids if i in labels]
    wins = total = 0
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            if labels[a] == labels[b]:
                continue
            total += 1
            # Higher rank should have label 1 to count as a win
            if labels[a] > labels[b]:
                wins += 1
    return wins/total if total else float('nan')

def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Load example sim predictions (campaign-level). For ad-level, we expect a file structure:
    # AELP2/reports/creative/<campaign_id>.json with items: {creative_id, sim_score, actual_label}
    creative_dir = REPORTS / 'creative'
    if not creative_dir.exists():
        (creative_dir).mkdir(parents=True, exist_ok=True)

    summary_rows = []
    out_summary = {
        'precision_at_5': None,
        'precision_at_10': None,
        'pairwise_win_rate': None,
        'kendall_tau': None,
        'n_campaigns': 0,
        'n_creatives': 0,
        'notes': []
    }

    p5_list, p10_list, pw_list, kt_list = [], [], [], []
    total_creatives = 0

    for file in sorted(creative_dir.glob('*.json')):
        data = load_json(file)
        items = data.get('items') or []
        if not items:
            continue
        # Build sim-ranked list and labels
        ranked = sorted(items, key=lambda x: (-(x.get('sim_score') or 0), x.get('creative_id')), )
        ranked_ids = [x['creative_id'] for x in ranked]
        labels = {x['creative_id']: 1 if x.get('actual_label') else 0 for x in items}
        total_creatives += len(ranked_ids)

        p5 = precision_at_k(ranked_ids, labels, 5)
        p10 = precision_at_k(ranked_ids, labels, 10)
        pw = pairwise_win_rate(ranked_ids, labels)
        # If we have a realized ranking metric (e.g., actual CAC), attempt Kendall tau
        realized_sorted = sorted(items, key=lambda x: (-(x.get('actual_score') or 0), x.get('creative_id')))
        realized_ids = [x['creative_id'] for x in realized_sorted]
        kt = kendall_tau(ranked_ids, realized_ids) if realized_ids else float('nan')

        summary_rows.append({
            'campaign_file': file.name,
            'n': len(items),
            'precision_at_5': p5,
            'precision_at_10': p10,
            'pairwise_win_rate': pw,
            'kendall_tau': kt,
        })
        for val, lst in [(p5,p5_list),(p10,p10_list),(pw,pw_list),(kt,kt_list)]:
            if isinstance(val, (int,float)) and not math.isnan(val):
                lst.append(val)

    out_summary['n_campaigns'] = len(summary_rows)
    out_summary['n_creatives'] = total_creatives
    if p5_list: out_summary['precision_at_5'] = round(sum(p5_list)/len(p5_list),4)
    if p10_list: out_summary['precision_at_10'] = round(sum(p10_list)/len(p10_list),4)
    if pw_list: out_summary['pairwise_win_rate'] = round(sum(pw_list)/len(pw_list),4)
    if kt_list: out_summary['kendall_tau'] = round(sum(kt_list)/len(kt_list),4)
    if not summary_rows:
        out_summary['notes'].append('No creative-level files found in AELP2/reports/creative/*.json; please export historical creative outcomes to enable ad-level metrics.')

    # Write JSON summary
    (REPORTS / 'ad_level_accuracy.json').write_text(json.dumps(out_summary, indent=2))

    # Write CSV detail
    with (REPORTS / 'ad_level_accuracy.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['campaign_file','n','precision_at_5','precision_at_10','pairwise_win_rate','kendall_tau'])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(json.dumps(out_summary, indent=2))

if __name__ == '__main__':
    main()

