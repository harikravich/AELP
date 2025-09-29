#!/usr/bin/env python3
from __future__ import annotations
"""
Fit isotonic calibrators mapping predicted CAC→actual CAC per campaign.

Inputs:
- AELP2/reports/weekly_creatives_cpc/*.json (must include `pred_cac_cpc`, `actual_score`, `test_spend`)

Outputs:
- AELP2/reports/calibration/campaign_<campaign_id>.json with sampled (x,y)
- AELP2/reports/calibration/summary.json with coverage stats
"""
import json, math
from pathlib import Path
from collections import defaultdict

def _req(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])
        return __import__(mod)

np = _req('numpy')
sk = _req('sklearn')
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives_cpc'
CAL = ROOT / 'AELP2' / 'reports' / 'calibration'
CAL.mkdir(parents=True, exist_ok=True)

def actual_cac(it: dict) -> float:
    s=float(it.get('test_spend') or 0.0); a=float(it.get('actual_score') or 0.0)
    return (s/a) if a>0 else float('inf')

def main():
    by_camp = defaultdict(list)
    for f in sorted(WK.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        cid=str(d.get('campaign_id') or f.name.split('_')[0])
        for it in items:
            p=it.get('pred_cac_cpc'); a=actual_cac(it)
            if p is None or not math.isfinite(a) or a<=0 or p<=0:
                continue
            if a>1e6 or p>1e6:
                continue
            by_camp[cid].append((float(p), float(a)))
    summary={'campaigns':[], 'total':0}
    for cid, pairs in by_camp.items():
        if len(pairs) < 15:
            continue
        xs = np.array([p for p,_ in pairs])
        ys = np.array([a for _,a in pairs])
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        _ = iso.fit_transform(xs, ys)
        grid = np.quantile(xs, np.linspace(0.0,1.0,21))
        fx = grid.tolist(); fy = iso.predict(grid).tolist()
        out = {'campaign_id': cid, 'n': len(pairs), 'x': fx, 'y': fy}
        (CAL / f'campaign_{cid}.json').write_text(json.dumps(out, indent=2))
        summary['campaigns'].append({'campaign_id': cid, 'n': len(pairs)})
        summary['total']+=len(pairs)
    (CAL / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
