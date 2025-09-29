#!/usr/bin/env python3
from __future__ import annotations
"""
Enrich weekly creatives with a CPC-mix based predicted CAC.

For each campaign-week file:
- Compute a campaign CVR prior from the previous W weeks (default 2): e_cvr_prev = sum(purchases)/sum(clicks).
- Load per-campaign creative aggregates from AELP2/reports/creative/<campaign_id>.json and compute cpc_hat for each creative
  as spend/clicks from placement_stats or `test_cpc` if available.
- For each item in the week file, set pred_cac_cpc = cpc_hat / max(e_cvr_prev, eps).
- Write to AELP2/reports/weekly_creatives_cpc/<campaign_id>_<week>.json

Env:
- AELP2_WEEKLY_DIR: source weekly dir (default AELP2/reports/weekly_creatives)
- AELP2_WK_CPC_OUT: output dir (default AELP2/reports/weekly_creatives_cpc)
- AELP2_WK_CVR_WEEKS: weeks to look back for CVR prior (default 2)
"""
import json, math, os
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
WK = Path(os.getenv('AELP2_WEEKLY_DIR') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives'))
OUT = Path(os.getenv('AELP2_WK_CPC_OUT') or (ROOT / 'AELP2' / 'reports' / 'weekly_creatives_cpc'))
OUT.mkdir(parents=True, exist_ok=True)
CRE = ROOT / 'AELP2' / 'reports' / 'creative'

def load_creative_cpc(campaign_id: str):
    p=CRE / f"{campaign_id}.json"
    if not p.exists():
        return {}
    d=json.loads(p.read_text()); items=d.get('items') or []
    cpc_map={}
    for it in items:
        cid=it.get('creative_id')
        cpc=it.get('test_cpc')
        if cpc is None:
            stats=it.get('placement_stats') or {}
            clicks=sum((v or {}).get('clicks',0.0) for v in stats.values())
            spend=sum((v or {}).get('spend',0.0) for v in stats.values())
            cpc=(spend/clicks) if clicks>0 else None
        if cpc is not None and math.isfinite(cpc):
            cpc_map[cid]=float(cpc)
    return cpc_map

def main():
    files=sorted(WK.glob('*.json'))
    by_c=defaultdict(list)
    for f in files:
        cid=f.name.split('_')[0]
        by_c[cid].append(f)
    lookback=int(os.getenv('AELP2_WK_CVR_WEEKS','2'))
    for cid, fl in by_c.items():
        fl=sorted(fl, key=lambda p: p.name.split('_')[1])
        cpc_map=load_creative_cpc(cid)
        # Precompute week-wise totals for CVR prior
        week_totals=[]
        for f in fl:
            d=json.loads(f.read_text()); items=d.get('items') or []
            clicks=sum(float(x.get('test_clicks') or 0.0) for x in items)
            purch=sum(float(x.get('actual_score') or 0.0) for x in items)
            week_totals.append({'file':f,'clicks':clicks,'purch':purch})
        for i,meta in enumerate(week_totals):
            # CVR prior from previous lookback weeks
            prev=week_totals[max(0,i-lookback):i]
            if prev:
                pc=sum(x['purch'] for x in prev); cc=sum(x['clicks'] for x in prev)
            else:
                # fallback: use current week totals to avoid zero
                pc=meta['purch']; cc=meta['clicks']
            e_cvr = (pc/cc) if cc>0 else 0.01
            e_cvr = float(max(1e-5, min(e_cvr, 0.5)))
            d=json.loads(meta['file'].read_text()); items=d.get('items') or []
            out_items=[]
            for it in items:
                cid_ad=it.get('creative_id')
                cpc = cpc_map.get(cid_ad)
                pred_cac_cpc = (cpc/e_cvr) if (cpc is not None and e_cvr>0) else None
                x=dict(it)
                if pred_cac_cpc is not None and math.isfinite(pred_cac_cpc):
                    x['pred_cac_cpc']=float(pred_cac_cpc)
                out_items.append(x)
            out={'campaign_id': d.get('campaign_id'), 'iso_week': d.get('iso_week'), 'items': out_items}
            (OUT/meta['file'].name).write_text(json.dumps(out, indent=2))
            print(f"wrote {OUT/meta['file'].name}")

if __name__=='__main__':
    main()

