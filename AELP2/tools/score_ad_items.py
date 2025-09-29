#!/usr/bin/env python3
from __future__ import annotations
"""
Score ad items using success_config proxies (best-effort on available fields).
Inputs: AELP2/competitive/ad_items_raw.json, AELP2/competitive/success_config.json
Outputs: AELP2/competitive/ad_items_scored.json
"""
import json, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'AELP2' / 'competitive' / 'ad_items_raw.json'
CFG = ROOT / 'AELP2' / 'competitive' / 'success_config.json'
OUT = ROOT / 'AELP2' / 'competitive' / 'ad_items_scored.json'

def parse_date(s: str | None):
    if not s: return None
    for fmt in ('%Y-%m-%dT%H:%M:%S%z','%Y-%m-%dT%H:%M:%S'):
        try: return dt.datetime.strptime(s, fmt)
        except Exception: pass
    return None

def main():
    items=(json.loads(RAW.read_text())['items'])
    cfg=json.loads(CFG.read_text())
    w=cfg['score_weights']
    proxies=cfg['proxies']
    today=dt.datetime.now(dt.timezone.utc)
    scored=[]
    for it in items:
        d=parse_date(it.get('created')) or today
        longevity=(today - d).days
        lon_norm=min(1.0, longevity / max(1, proxies['longevity_days_min']))
        variant_density=1.0 if it.get('title') and it.get('body') else 0.5  # proxy: has both fields
        place_geo=1.0 if (it.get('placements') or ['AUTOMATIC_FORMAT']) else 0.3
        refresh=0.5  # unknown
        creator=0.0  # unknown
        lp_reuse=0.5 if (it.get('link')) else 0.2
        score=(lon_norm*w['longevity'] + variant_density*w['variant_density'] + place_geo*w['placement_geo'] + refresh*w['refresh_velocity'] + creator*w['creator'] + lp_reuse*w['lp_reuse'])
        it2={**it, 'longevity_days': longevity, 'score': round(score,3)}
        scored.append(it2)
    OUT.write_text(json.dumps({'items': scored}, indent=2))
    print(json.dumps({'scored': len(scored), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

