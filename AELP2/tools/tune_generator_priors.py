#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUTD = ROOT / 'AELP2' / 'priors'; OUTD.mkdir(parents=True, exist_ok=True)

def main():
    fmt_counts = {'9x16':0, '1x1-4x5':0, 'video':0, 'unknown':0}
    tags = {'balance_parental':0, 'breach_alerts':0, 'speed_benefit':0}
    total_pos = 0
    # Fallback weighting when positives are sparse: use top-K by sim_score
    fallback_scores = []
    for f in ENR.glob('*.json'):
        data = json.loads(f.read_text())
        items = (data.get('items') or [])
        for it in items:
            if int(it.get('actual_label',0)) != 1: continue
            total_pos += 1
            dr = (it.get('dna_real') or {})
            fmt_counts[dr.get('ratio_tag','unknown')] = fmt_counts.get(dr.get('ratio_tag','unknown'),0)+1
            for t in (it.get('dna') or {}).get('tags', []):
                if t in tags: tags[t]+=1
        # collect fallback candidate weights from top-K sim_score
        items_sorted = sorted(items, key=lambda x: float(x.get('sim_score') or 0.0), reverse=True)[:5]
        for it in items_sorted:
            fallback_scores.append(it)
    # convert to weights
    def normalize(d):
        s = sum(d.values()) or 1
        return {k: round(v/s,4) for k,v in d.items()}
    if total_pos == 0 and fallback_scores:
        # Derive weak priors from top candidates
        for it in fallback_scores:
            dr = (it.get('dna_real') or {})
            fmt_counts[dr.get('ratio_tag','unknown')] = fmt_counts.get(dr.get('ratio_tag','unknown'),0)+1
        # tags remain zero if absent
    priors = {
        'formats': normalize(fmt_counts),
        'tags': normalize(tags)
    }
    (OUTD / 'generator_priors.json').write_text(json.dumps(priors, indent=2))
    print(json.dumps(priors, indent=2))

if __name__ == '__main__':
    main()
