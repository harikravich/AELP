#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'AELP2' / 'reports'

def loadp(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def metric(d, k):
    v = d.get(k)
    return v if isinstance(v, (int,float)) else float('nan')

def main():
    baseline = loadp(R / 'ad_level_accuracy.json')
    v22 = loadp(R / 'ad_level_accuracy_v22.json')
    v23 = loadp(R / 'ad_level_accuracy_v23.json')
    v24 = loadp(R / 'ad_level_accuracy_v24.json')
    options = []
    for name, d in [('baseline', baseline), ('v22', v22), ('v23', v23), ('v24', v24)]:
        if not d: continue
        options.append({
            'name': name,
            'p5': metric(d, 'precision_at_5'),
            'p10': metric(d, 'precision_at_10'),
            'pairwise': metric(d, 'pairwise_win_rate')
        })
    # choose best by p5 then pairwise
    best = None
    for opt in options:
        if best is None:
            best = opt
            continue
        if (opt['p5'] > best['p5']) or (opt['p5'] == best['p5'] and (opt['pairwise'] or 0) > (best['pairwise'] or 0)):
            best = opt
    choice = {'selector': best['name'] if best else 'baseline', 'candidates': options}
    (R / 'selector_choice.json').write_text(json.dumps(choice, indent=2))
    print(json.dumps(choice, indent=2))

if __name__ == '__main__':
    main()
