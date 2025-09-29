#!/usr/bin/env python3
from __future__ import annotations
"""
Check dual-gate precision and yield against thresholds and write a status file.

Inputs: AELP2/reports/dual_gate_weekly.json
Env:
- AELP2_DUAL_MIN_PREC (default 0.25)
- AELP2_DUAL_MIN_YIELD (default 0.01)

Outputs: AELP2/reports/dual_gate_status.json
"""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'AELP2' / 'reports' / 'dual_gate_weekly.json'
OUT = ROOT / 'AELP2' / 'reports' / 'dual_gate_status.json'

def main():
    if not SRC.exists():
        OUT.write_text(json.dumps({'status':'missing'}, indent=2)); print('{"status":"missing"}'); return
    d=json.loads(SRC.read_text()); grid=d.get('grid') or {}
    minp=float(os.getenv('AELP2_DUAL_MIN_PREC','0.25'))
    miny=float(os.getenv('AELP2_DUAL_MIN_YIELD','0.01'))
    ok=[]; fail=[]
    for k,v in grid.items():
        dp=v.get('dual_precision') or 0.0; y=v.get('yield') or 0.0
        (ok if (dp>=minp and y>=miny) else fail).append({'key':k,'dual_precision':dp,'yield':y})
    status={'ok': ok, 'fail': fail, 'min_precision': minp, 'min_yield': miny, 'ts': __import__('time').time()}
    OUT.write_text(json.dumps(status, indent=2))
    print(json.dumps({'ok': len(ok), 'fail': len(fail)}, indent=2))

if __name__=='__main__':
    main()

