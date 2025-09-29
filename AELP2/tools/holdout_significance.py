#!/usr/bin/env python3
from __future__ import annotations
"""
Bootstrap significance test for CAC differences between slates using reference CACs.

Inputs: us_cac_volume_forecasts.json, rl_offline_simulation.json,
        auctiongym_offline_simulation_calibrated.json, recsim_offline_simulation.json
Output: AELP2/reports/holdout_significance.json
"""
import json, random
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
RL = ROOT / 'AELP2' / 'reports' / 'rl_offline_simulation.json'
AU = ROOT / 'AELP2' / 'reports' / 'auctiongym_offline_simulation_calibrated.json'
RS = ROOT / 'AELP2' / 'reports' / 'recsim_offline_simulation.json'
OUT = ROOT / 'AELP2' / 'reports' / 'holdout_significance.json'

def load_ref():
    d = json.loads(REF.read_text())
    ref = {}
    for r in d['items']:
        bud = r['budgets']['30000'] if '30000' in r['budgets'] else next(iter(r['budgets'].values()))
        ref[r['creative_id']] = float(bud['cac']['p50'])
    return ref

def load_slate_ts():
    d = json.loads(RL.read_text())
    return [a['id'] for a in d.get('ranking', [])[:8]]

def load_slate_au():
    d = json.loads(AU.read_text())
    items = sorted(d['items'], key=lambda x: (x['cac'] if x['cac'] is not None else 1e9))
    return [x['creative_id'] for x in items[:8]]

def load_slate_rs():
    d = json.loads(RS.read_text())
    return d.get('slate', [])[:8]

def bootstrap_delta(ref, slate_a, slate_b, B=2000, seed=0):
    random.seed(seed)
    a = [ref[c] for c in slate_a if c in ref]
    b = [ref[c] for c in slate_b if c in ref]
    n = min(len(a), len(b))
    deltas = []
    for _ in range(B):
        sa = [random.choice(a) for _ in range(n)]
        sb = [random.choice(b) for _ in range(n)]
        deltas.append(mean(sb) - mean(sa))  # positive means B worse (higher CAC)
    deltas.sort()
    ci = (deltas[int(0.025*B)], deltas[int(0.975*B)])
    return {'delta_mean': round(mean(deltas),2), 'ci95': [round(ci[0],2), round(ci[1],2)]}

def main():
    ref = load_ref()
    ts = load_slate_ts()
    au = load_slate_au()
    rs = load_slate_rs()
    out = {
        'ts_vs_au': bootstrap_delta(ref, au, ts),
        'ts_vs_rs': bootstrap_delta(ref, rs, ts),
        'au_vs_rs': bootstrap_delta(ref, rs, au)
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()

