#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / 'AELP2' / 'reports' / 'auction_backtest_summary.json'
OUT = ROOT / 'AELP2' / 'reports' / 'auction_backtest_precision_ci.json'

def main():
    d = json.loads(INP.read_text())
    vals = [it['p10'] for it in d.get('details') or [] if isinstance(it.get('p10'), (int,float))]
    if not vals:
        OUT.write_text(json.dumps({'status':'no_data'}, indent=2)); print('{"status":"no_data"}') ; return
    B=2000; deltas=[]; random.seed(0)
    n=len(vals)
    boots=[mean([random.choice(vals) for _ in range(n)]) for _ in range(B)]
    boots.sort()
    ci=(boots[int(0.025*B)], boots[int(0.975*B)])
    out={'mean': round(mean(vals),3), 'ci95': [round(ci[0],3), round(ci[1],3)], 'n_campaigns': n}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()

