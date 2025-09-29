#!/usr/bin/env python3
from __future__ import annotations
"""
Compute weekly (7-day aggregated) fidelity from sim_fidelity_campaigns_temporal_v3.json.
Aggregates actual and predicted purchases across the last 7 days and reports weekly relative error and interval coverage.
Output: AELP2/reports/weekly_fidelity.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'AELP2' / 'reports' / 'sim_fidelity_campaigns_temporal_v3.json'
OUT = ROOT / 'AELP2' / 'reports' / 'weekly_fidelity.json'

def main():
    if not SIM.exists():
        print('missing sim v3 json'); return
    d=json.loads(SIM.read_text())
    daily=d.get('daily') or []
    if not daily:
        print('no daily rows'); return
    act=sum(float(r.get('actual_purch') or 0.0) for r in daily)
    pred_med=sum(float(r.get('pred_purch_med') or 0.0) for r in daily)
    pred_lo=sum(float(r.get('pred_purch_p10') or 0.0) for r in daily)
    pred_hi=sum(float(r.get('pred_purch_p90') or 0.0) for r in daily)
    rel_err = (abs(pred_med-act)/act) if act>0 else None
    covered = (act>=pred_lo and act<=pred_hi)
    out={'weekly': {'actual_purch_total': act,
                    'pred_purch_med_total': pred_med,
                    'pred_purch_p10_total': pred_lo,
                    'pred_purch_p90_total': pred_hi,
                    'relative_error': (round(rel_err*100,2) if rel_err is not None else None),
                    'covered80': covered}}
    OUT.write_text(json.dumps(out, indent=2))
    print('AELP2/reports/weekly_fidelity.json')

if __name__=='__main__':
    main()

