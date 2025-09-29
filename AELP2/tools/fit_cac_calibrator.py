#!/usr/bin/env python3
from __future__ import annotations
"""
Fit a monotonic calibrator mapping predicted CAC to actual CAC using isotonic regression.

Inputs:
- AELP2/reports/weekly_creatives_cpc/*.json (pred_cac_cpc preferred), else use weekly_creatives/*.json with spend/sim proxy.

Outputs:
- AELP2/reports/cac_calibrator.json with fields:
  { 'x': [...], 'y': [...], 'note': 'isotonic (increasing)', 'n': N }

Env:
- AELP2_WEEKLY_DIR: override weekly dir used to collect samples
"""
import json, math, os
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
WK_DEFAULT = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
WK_CPC = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_cpc'
OUT = ROOT / 'AELP2' / 'reports' / 'cac_calibrator.json'

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def pred_cac(it: dict) -> float:
    if it.get('pred_cac_cpc') is not None:
        try:
            return float(it.get('pred_cac_cpc'))
        except Exception:
            pass
    s=float(it.get('test_spend') or 0.0); sim=float(it.get('sim_score') or 0.0)
    return (s/sim) if sim>0 else float('inf')

def main():
    wk = Path(os.getenv('AELP2_WEEKLY_DIR') or (WK_CPC if WK_CPC.exists() else WK_DEFAULT))
    xs=[]; ys=[]
    for f in sorted(wk.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        for it in items:
            a=actual_cac(it); p=pred_cac(it)
            if math.isfinite(a) and math.isfinite(p) and p>0 and a>0 and a<1e6 and p<1e6:
                xs.append(float(p)); ys.append(float(a))
    if len(xs) < 50:
        OUT.write_text(json.dumps({'note':'insufficient data','n':len(xs)}, indent=2))
        print(json.dumps({'n':len(xs),'status':'insufficient'}, indent=2)); return
    xs=np.array(xs); ys=np.array(ys)
    # Fit isotonic increasing function
    iso=IsotonicRegression(increasing=True, out_of_bounds='clip')
    yhat=iso.fit_transform(xs, ys)
    # Export as sampled points across the support
    grid=np.quantile(xs, np.linspace(0.0,1.0,21))
    fx=grid.tolist(); fy=iso.predict(grid).tolist()
    OUT.write_text(json.dumps({'x':fx,'y':fy,'note':'isotonic_increasing','n':len(xs)}, indent=2))
    print(json.dumps({'n':len(xs),'points':len(fx)}, indent=2))

if __name__=='__main__':
    main()

