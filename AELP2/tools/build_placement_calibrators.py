#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / 'AELP2' / 'reports' / 'placement_conversions'
OUT = ROOT / 'AELP2' / 'reports'

def main():
    clicks=0.0; purch=0.0; agg={}
    for f in INP.glob('*.json'):
        d=json.loads(f.read_text()); data=d.get('data') or {}
        for k,v in data.items():
            c=float(v.get('clicks',0.0)); p=float(v.get('purch',0.0))
            if k not in agg: agg[k]={'clicks':0.0,'purch':0.0}
            agg[k]['clicks']+=c; agg[k]['purch']+=p
            clicks+=c; purch+=p
    g_rate=(purch/clicks) if clicks>0 else 0.01
    calibrators={}
    for k,v in agg.items():
        rate=(v['purch']/v['clicks']) if v['clicks']>0 else g_rate
        calibrators[k]= {'ctr': None, 'cvr': float(rate/g_rate) if g_rate>0 else 1.0}
    (OUT/'placement_calibrators.json').write_text(json.dumps({'global_cvr': g_rate, 'calibrators': calibrators}, indent=2))
    print('wrote placement_calibrators.json')

if __name__=='__main__':
    main()

