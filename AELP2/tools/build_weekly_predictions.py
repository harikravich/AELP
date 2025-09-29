#!/usr/bin/env python3
from __future__ import annotations
"""
Build weekly (calendar) predictions and actuals per creative from ad-level daily insights.
For each campaign and ISO week t, use prior 2 weeks (t-2,t-1) to estimate a CVR prior and predict purchases_t = clicks_t * E[cvr_prior].
Also compute actual purchases_t and CAC_t. Exports per-campaign weekly JSON files for WBUA evaluation.
Outputs: AELP2/reports/weekly_creatives/<campaign_id>_<iso_year>W<iso_week>.json
"""
import json, math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'AELP2' / 'raw' / 'ad_daily'
OUT = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
OUT.mkdir(parents=True, exist_ok=True)

def iso_year_week(d: str):
    y,m,dd = map(int, d.split('-'))
    iso = datetime(y,m,dd).isocalendar()
    return iso.year, iso.week

def main():
    # Load all shards into memory (chunked)
    by_camp_day = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    # structure: by_camp_day[campaign_id][date]['clicks'/'purch'/'spend'] accumulates per ad_id: also keep per-ad mapping for weekly
    per_ad_day = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    # per_ad_day[campaign_id][ad_id][date]['clicks'/'purch'/'spend']
    # ad meta (stable per ad): adset_id, dominant placement (best-effort)
    ad_meta = defaultdict(lambda: defaultdict(dict))  # ad_meta[campaign_id][ad_id] -> {'adset_id': str, 'placement': str}
    for shard in sorted(RAW.glob('*.jsonl')):
        for line in shard.read_text().splitlines():
            if not line.strip(): continue
            try:
                r=json.loads(line)
            except Exception:
                continue
            cid = r.get('campaign_id') or 'unknown'
            ad = r.get('ad_id') or 'unknown'
            d = r.get('date_start')
            if not cid or not ad or not d: continue
            clicks = float(r.get('clicks') or 0.0)
            spend = float(r.get('spend') or 0.0)
            purch = 0.0
            for a in (r.get('actions') or []):
                if a.get('action_type')=='offsite_conversion.fb_pixel_purchase':
                    try: purch=float(a.get('value') or 0.0)
                    except Exception: purch=0.0
            by_camp_day[cid][d]['clicks'] += clicks
            by_camp_day[cid][d]['purch'] += purch
            by_camp_day[cid][d]['spend'] += spend
            per_ad_day[cid][ad][d]['clicks'] += clicks
            per_ad_day[cid][ad][d]['purch'] += purch
            per_ad_day[cid][ad][d]['spend'] += spend
            # meta fields
            adset_id = r.get('adset_id') or ad_meta[cid].get(ad, {}).get('adset_id') or None
            # Try to infer a dominant placement key if present in record; fall back to 'unknown'
            placement = None
            # Some exports may include publisher_platform or placement breakdowns; keep the first seen non-empty
            for k in ('publisher_platform', 'platform_position', 'device_platform'):
                v = r.get(k)
                if v:
                    placement = str(v)
                    break
            prev_meta = ad_meta[cid].get(ad, {})
            if adset_id and not prev_meta.get('adset_id'):
                prev_meta['adset_id'] = adset_id
            if placement and not prev_meta.get('placement'):
                prev_meta['placement'] = placement
            ad_meta[cid][ad] = prev_meta

    # Build weekly per ad predictions
    for cid, ads in per_ad_day.items():
        # collect all dates
        all_dates = sorted({d for ad_map in ads.values() for d in ad_map.keys()})
        # group by iso week
        weeks = sorted({iso_year_week(d) for d in all_dates})
        # build map from week -> dates
        week_dates = defaultdict(list)
        for d in all_dates:
            week_dates[iso_year_week(d)].append(d)
        # iterate weeks
        for i, wk in enumerate(weeks):
            # training weeks: i-2,i-1
            if i<2: continue
            train_weeks = [weeks[i-2], weeks[i-1]]
            test_week = wk
            # compute prior from train weeks
            tr_clicks = 0.0; tr_purch = 0.0
            for tw in train_weeks:
                for d in week_dates[tw]:
                    tr_clicks += by_camp_day[cid][d]['clicks']
                    tr_purch  += by_camp_day[cid][d]['purch']
            # Beta prior
            k=150.0
            rate=(tr_purch/tr_clicks) if tr_clicks>0 else 0.01
            rate = max(1e-5, min(rate, 0.5))
            alpha0, beta0 = rate*k, (1-rate)*k
            # build items for all ads with clicks in test week
            items=[]
            tot_week_clicks = 0.0
            for ad, daymap in ads.items():
                clk=sum(daymap[d]['clicks'] for d in week_dates[test_week] if d in daymap)
                sp = sum(daymap[d]['spend'] for d in week_dates[test_week] if d in daymap)
                pu = sum(daymap[d]['purch'] for d in week_dates[test_week] if d in daymap)
                if clk<=0 and sp<=0 and pu<=0: continue
                e_cvr = alpha0/(alpha0+beta0)
                pred_purch = clk * e_cvr
                actual_cac = (sp/pu) if pu>0 else float('inf')
                meta = ad_meta[cid].get(ad, {})
                items.append({
                    'creative_id': ad,
                    'adset_id': meta.get('adset_id'),
                    'placement': meta.get('placement') or 'unknown',
                    'sim_score': float(pred_purch),
                    'actual_score': float(pu),
                    'actual_cac': actual_cac,
                    'test_clicks': float(clk),
                    'test_spend': float(sp)
                })
                tot_week_clicks += clk
            if not items:
                continue
            out={'campaign_id': cid, 'iso_week': f"{test_week[0]}W{test_week[1]:02d}", 'items': items}
            (OUT / f"{cid}_{test_week[0]}W{test_week[1]:02d}.json").write_text(json.dumps(out, indent=2))
            print(f"wrote {cid}_{test_week[0]}W{test_week[1]:02d}.json ({len(items)} ads)")

if __name__=='__main__':
    main()
