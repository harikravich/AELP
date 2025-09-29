#!/usr/bin/env python3
from __future__ import annotations
"""
Holdout-style evaluation for Auction vs Thompson vs RecSim slates using available reports.

Method (pragmatic, given available artifacts):
- Treat us_cac_volume_forecasts.json p50 as the current best reference for per-creative CAC/signups at $30k (proxy for "observed" on holdout slice).
- For each candidate slate:
  * TS: take the top-8 from rl_offline_simulation.json ranking
  * AU: take the top-8 (lowest calibrated CAC) from auctiongym_offline_simulation_calibrated.json
  * RS: take the top-8 from recsim_offline_simulation.json
- Compute per-slate metrics against the reference:
  * avg_cac_p50, sum_signups_p50 (from reference)
  * precision@10 proxy: overlap with the top-10 by reference CAC
  * calibration proxy: mean |pred_cac - ref_cac| / ref_cac for creatives present in each slate
    - TS predicted CAC = ref cac (baseline forecast, idealized)
    - AU predicted CAC = calibrated CAC from auction file
    - RS predicted CAC = cac_adj scaled to $30k (already in units of CAC) from recsim file
Outputs: AELP2/reports/holdout_evaluation.json
"""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
RL = ROOT / 'AELP2' / 'reports' / 'rl_offline_simulation.json'
AU = ROOT / 'AELP2' / 'reports' / 'auctiongym_offline_simulation_calibrated.json'
RS = ROOT / 'AELP2' / 'reports' / 'recsim_offline_simulation.json'
OUT = ROOT / 'AELP2' / 'reports' / 'holdout_evaluation.json'

def load_ref():
    d = json.loads(REF.read_text())
    ref = {}
    for r in d['items']:
        bud = r['budgets']['30000'] if '30000' in r['budgets'] else next(iter(r['budgets'].values()))
        ref[r['creative_id']] = {
            'cac': float(bud['cac']['p50']),
            'signups': float(bud['signups']['p50'])
        }
    # top-10 by lowest CAC
    top10 = [cid for cid, _ in sorted(((cid, v['cac']) for cid,v in ref.items()), key=lambda t: t[1])[:10]]
    return ref, top10

def load_ts():
    d = json.loads(RL.read_text())
    return [a['id'] for a in d.get('ranking', [])[:8]]

def load_au():
    d = json.loads(AU.read_text())
    items = sorted(d['items'], key=lambda x: (x['cac'] if x['cac'] is not None else 1e9))
    return [x['creative_id'] for x in items[:8]], {x['creative_id']: x['cac'] for x in d['items']}

def load_rs():
    d = json.loads(RS.read_text())
    return d.get('slate', [])[:8], {x['creative_id']: x['cac_adj'] for x in d['items']}

def slate_stats(name, slate, ref, top10, preds=None):
    cacs = [ref[c]['cac'] for c in slate if c in ref]
    su = [ref[c]['signups'] for c in slate if c in ref]
    p10 = len(set(slate).intersection(top10)) / min(10, len(slate))
    # calibration proxy
    cal = None
    if preds:
        diffs = []
        for c in slate:
            if c in ref and c in preds and preds[c] is not None:
                diffs.append(abs(preds[c] - ref[c]['cac'])/max(ref[c]['cac'], 1e-6))
        if diffs:
            cal = mean(diffs)
    return {
        'avg_cac_p50': round(mean(cacs),2) if cacs else None,
        'sum_signups_p50': round(sum(su),1) if su else None,
        'precision_at_10_proxy': round(p10,3),
        'calibration_mape': (round(cal,3) if cal is not None else None),
        'n': len(cacs)
    }

def main():
    ref, top10 = load_ref()
    ts = load_ts()
    au, au_preds = load_au()
    rs, rs_preds = load_rs()
    out = {
        'reference_top10': top10,
        'slates': {
            'thompson': {'ids': ts, 'metrics': slate_stats('ts', ts, ref, top10)},
            'auction': {'ids': au, 'metrics': slate_stats('au', au, ref, top10, au_preds)},
            'recsim': {'ids': rs, 'metrics': slate_stats('rs', rs, ref, top10, rs_preds)},
        }
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out['slates'], indent=2))

if __name__ == '__main__':
    main()

