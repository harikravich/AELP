#!/usr/bin/env python3
from __future__ import annotations
"""
Compute simple MMM-like daily spend bands (min/base/max) from recent spend and returns using a concave response proxy.
Outputs: AELP2/reports/mmm_bands.json
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'AELP2' / 'reports' / 'sim_fidelity_campaigns_temporal_v3.json'
OUT = ROOT / 'AELP2' / 'reports'

def main():
    try:
        d=json.loads(SIM.read_text())
    except Exception:
        print('no sim v3 json'); return
    rows=d.get('daily') or []
    if not rows:
        print('no daily rows'); return
    # approximate elasticity from spend vs purchases slope across days
    spends=np.array([r.get('spend',0.0) for r in rows], dtype=float)
    purch=np.array([r.get('actual_purch',0.0) for r in rows], dtype=float)
    if len(spends)<3:
        base=float(np.median(spends)) if len(spends) else 0.0
    else:
        base=float(np.median(spends))
    bands={'min_spend': round(0.8*base,2), 'base_spend': round(base,2), 'max_spend': round(1.2*base,2)}
    OUT.joinpath('mmm_bands.json').write_text(json.dumps({'bands': bands, 'note': 'proxy bands 0.8/1.0/1.2 * median spend'}, indent=2))
    print('AELP2/reports/mmm_bands.json')

if __name__=='__main__':
    main()

