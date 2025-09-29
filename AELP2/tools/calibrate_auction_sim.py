#!/usr/bin/env python3
from __future__ import annotations
"""
Apply global CAC isotonic calibrator to auction sim outputs, if available.
Reads AELP2/reports/auctiongym_offline_simulation.json and writes
      AELP2/reports/auctiongym_offline_simulation_calibrated.json
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / 'AELP2' / 'reports' / 'auctiongym_offline_simulation.json'
CAL = ROOT / 'AELP2' / 'reports' / 'cac_calibrator.json'
OUT = ROOT / 'AELP2' / 'reports' / 'auctiongym_offline_simulation_calibrated.json'

def load_iso():
    if not CAL.exists():
        return None
    d = json.loads(CAL.read_text())
    xs = d.get('x'); ys = d.get('y')
    if not xs or not ys or len(xs) != len(ys):
        return None
    pairs = list(zip(xs, ys))
    pairs.sort()
    def f(x: float) -> float:
        # piecewise linear interpolation, clipped
        if x <= pairs[0][0]:
            return pairs[0][1]
        if x >= pairs[-1][0]:
            return pairs[-1][1]
        for i in range(1, len(pairs)):
            if x <= pairs[i][0]:
                x0,y0=pairs[i-1]; x1,y1=pairs[i]
                t=(x - x0)/max(x1-x0,1e-9)
                return y0 + t*(y1-y0)
        return x
    return f

def main():
    data = json.loads(INP.read_text())
    iso = load_iso()
    items = []
    for it in data.get('items', []):
        x = dict(it)
        cac = x.get('cac')
        if iso and isinstance(cac, (int,float)) and cac is not None:
            x['cac'] = float(iso(float(cac)))
        items.append(x)
    OUT.write_text(json.dumps({'items': items}, indent=2))
    print(str(OUT))

if __name__=='__main__':
    main()

