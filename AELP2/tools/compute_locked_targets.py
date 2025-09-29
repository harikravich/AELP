#!/usr/bin/env python3
from __future__ import annotations
"""
Compute locked target_CAC per campaign from recent weekly files to stabilize gates
for the first policy-compliant period.

Logic:
- Read weekly files (policy dir if present), take the latest N weeks (default 2).
- For each campaign, compute the median CAC across those weeks (per-week medians, then median-of-medians).
- Write AELP2/reports/target_cac_locked.json: {campaign_id: target_cac}

Env:
- AELP2_WEEKLY_DIR: override weekly dir (default AELP2/reports/weekly_creatives)
- AELP2_LOCK_WEEKS: number of recent weeks to aggregate (default 2)
"""
import json, math, os
from pathlib import Path
from statistics import median
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives'))
OUT = ROOT / 'AELP2' / 'reports' / 'target_cac_locked.json'

def actual_cac(it: dict) -> float:
    p=float(it.get('actual_score') or 0.0); s=float(it.get('test_spend') or 0.0)
    return (s/p) if p>0 else float('inf')

def main():
    n_weeks = int(os.getenv('AELP2_LOCK_WEEKS','2'))
    files = sorted(WK.glob('*.json'))
    by_c = defaultdict(list)
    for f in files:
        cid=f.name.split('_')[0]
        by_c[cid].append(f)
    locked={}
    for cid, fl in by_c.items():
        fl=sorted(fl, key=lambda p: p.name.split('_')[1])[-n_weeks:]
        week_meds=[]
        for f in fl:
            d=json.loads(f.read_text()); items=d.get('items') or []
            cacs=[actual_cac(it) for it in items if math.isfinite(actual_cac(it))]
            if cacs:
                week_meds.append(median(cacs))
        if week_meds:
            locked[cid]=float(median(week_meds))
    OUT.write_text(json.dumps({'targets': locked, 'weeks': n_weeks}, indent=2))
    print(json.dumps({'campaigns': len(locked)}, indent=2))

if __name__=='__main__':
    main()

