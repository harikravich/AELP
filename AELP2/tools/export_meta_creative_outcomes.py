#!/usr/bin/env python3
"""
Export historical per-ad outcomes from Meta (read-only) and build files for ad-level accuracy.

For each campaign, we compute:
  - Aggregated actuals per ad_id over the last 7 test days (clicks, spend, purchases, CAC)
  - A simple per-ad simulated score using a campaign-level CVR prior estimated from the preceding 14 train days

Outputs one file per campaign:
  AELP2/reports/creative/<campaign_id>.json with structure:
  {
    "campaign_id": "...",
    "items": [
      {"creative_id": "<ad_id>", "sim_score": <float>, "actual_label": 0|1, "actual_score": <float>}
    ]
  }

Notes:
  - This uses raw Graph API via requests; requires META_ACCESS_TOKEN and META_ACCOUNT_ID in .env or env.
  - "actual_label" uses a conservative rule: label 1 if CAC <= campaign median CAC and purchases >= campaign median purchases among ads.
"""
from __future__ import annotations
import os, json, math
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / 'AELP2' / 'reports' / 'creative'
OUTDIR.mkdir(parents=True, exist_ok=True)

META_BASE = "https://graph.facebook.com/v21.0"


def read_env() -> Tuple[str, str]:
    tok = os.getenv('META_ACCESS_TOKEN') or os.getenv('META_ACCESS_TOKEN_DISABLED')
    acct = os.getenv('META_ACCOUNT_ID')
    if not (tok and acct):
        envp = ROOT / '.env'
        if envp.exists():
            for ln in envp.read_text().splitlines():
                if ln.startswith('export META_ACCESS_TOKEN=') and not tok:
                    tok = ln.split('=',1)[1].strip()
                if ln.startswith('export META_ACCESS_TOKEN_DISABLED=') and not tok:
                    tok = ln.split('=',1)[1].strip()
                if ln.startswith('export META_ACCOUNT_ID=') and not acct:
                    acct = ln.split('=',1)[1].strip()
    if not tok or not acct:
        raise RuntimeError('Missing META_ACCESS_TOKEN or META_ACCOUNT_ID')
    return tok, acct


def get_insights_daily_level_ad(token: str, account_id: str, since: str, until: str) -> List[dict]:
    url = f"{META_BASE}/{account_id}/insights"
    base_url = f"{META_BASE}/{account_id}/insights"
    def make_params(use_breakdowns: bool) -> Dict[str, str]:
        p = {
            'time_increment': 1,
            'level': 'ad',
            'fields': 'ad_id,ad_name,adset_id,campaign_id,date_start,impressions,clicks,spend,frequency,actions',
            'time_range': json.dumps({'since': since, 'until': until}),
            'access_token': token,
            'limit': 500,
        }
        if use_breakdowns:
            p['breakdowns'] = 'publisher_platform,platform_position,impression_device'
        return p
    out: List[dict] = []
    pages = 0
    use_breakdowns = True
    while True:
        # Use params only on the first request of the current paging sequence
        r = requests.get(url, params=make_params(use_breakdowns) if pages == 0 else None, timeout=120)
        if r.status_code >= 400:
            if use_breakdowns:
                # Restart without breakdowns from the beginning
                use_breakdowns = False
                pages = 0
                url = base_url
                time.sleep(0.5)
                continue
            r.raise_for_status()
        js = r.json()
        out.extend(js.get('data', []))
        nxt = (js.get('paging') or {}).get('next')
        if not nxt:
            break
        url = nxt
        pages += 1
        if pages % 50 == 0:
            time.sleep(0.5)
        if pages > 4000:
            break
    return out


def extract_purchases(actions: List[dict] | None) -> float:
    if not actions:
        return 0.0
    for a in actions:
        if a.get('action_type') == 'offsite_conversion.fb_pixel_purchase':
            try:
                return float(a.get('value') or 0)
            except Exception:
                return 0.0
    return 0.0


def main():
    token, acct = read_env()
    today = date.today()
    until = today - timedelta(days=1)
    window_days = int(os.getenv('AELP2_WINDOW_DAYS', '28'))
    if window_days < 7:
        window_days = 7
    since = until - timedelta(days=window_days - 1)

    rows = get_insights_daily_level_ad(token, acct, since.isoformat(), until.isoformat())
    # organize by campaign, by day, by ad
    by_campaign: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        camp = r.get('campaign_id') or 'unknown'
        d = r.get('date_start')
        ad = r.get('ad_id') or 'unknown'
        adset = r.get('adset_id') or 'unknown'
        clicks = float(r.get('clicks') or 0)
        spend = float(r.get('spend') or 0)
        purch = extract_purchases(r.get('actions'))
        # Placement/device breakdown key
        pp = r.get('publisher_platform') or 'unknown'
        pos = r.get('platform_position') or 'unknown'
        dev = r.get('impression_device') or 'unknown'
        key = f"{pp}|{pos}|{dev}"
        # Initialize or update the entry
        entry = by_campaign[camp][d].get(ad)
        if not entry:
            entry = by_campaign[camp][d][ad] = {
                'clicks': 0.0,
                'spend': 0.0,
                'purch': 0.0,
                'freq': 0.0,
                'ad_name': r.get('ad_name'),
                'adset_id': adset,
                'placement_mix': {}
            }
        entry['clicks'] += clicks
        entry['spend'] += spend
        entry['purch'] += purch
        entry['freq'] = max(entry['freq'], float(r.get('frequency') or entry['freq']))
        mix = entry['placement_mix']
        if key not in mix:
            mix[key] = {'clicks': 0.0, 'spend': 0.0, 'impr': 0.0}
        mix[key]['clicks'] += clicks
        mix[key]['spend'] += spend
        mix[key]['impr'] += float(r.get('impressions') or 0)
    all_days = sorted({d for camp in by_campaign.values() for d in camp.keys()})
    if len(all_days) < 7:
        print('Not enough days to compute windows')
        return
    # define windows via env: train N → test T (defaults 21→7)
    TEST_DAYS = int(os.getenv('AELP2_TEST_DAYS', '7'))
    TRAIN_DAYS = int(os.getenv('AELP2_TRAIN_DAYS', '21'))
    test_days = all_days[-TEST_DAYS:]
    train_days = all_days[-(TEST_DAYS + TRAIN_DAYS):-TEST_DAYS] if len(all_days) >= (TEST_DAYS + 1) else all_days[:-TEST_DAYS]

    # Label thresholds (env-tunable)
    MIN_PURCH = int(os.getenv('AELP2_LABEL_MIN_PURCH', '3'))
    MIN_CLICKS = int(os.getenv('AELP2_LABEL_MIN_CLICKS', '50'))
    MIN_SPEND = float(os.getenv('AELP2_LABEL_MIN_SPEND', '50'))

    for camp_id, daily in by_campaign.items():
        # estimate campaign-level CVR prior from train window
        train_clicks = 0.0
        train_purch = 0.0
        for d in train_days:
            if d not in daily: continue
            for ad, rec in daily[d].items():
                train_clicks += rec['clicks']
                train_purch += rec['purch']
        # Beta prior with modest strength
        rate = (train_purch/train_clicks) if train_clicks > 0 else 0.01
        rate = max(1e-5, min(rate, 0.5))
        k = float(os.getenv('AELP2_PRIOR_STRENGTH_K', '100.0'))
        alpha0, beta0 = rate*k, (1-rate)*k

        # aggregate per-ad actuals over test window and compute a simple simulated score
        per_ad = defaultdict(lambda: {'clicks':0.0,'spend':0.0,'purch':0.0,'placement_mix':{},'placement_stats':{},
                                      'ad_name':None,'adset_id':None,'train_clicks':0.0,'train_purch':0.0,
                                      'freq_series':[]})
        # accumulate train stats per ad for hierarchical prior
        for d in train_days:
            if d not in daily: continue
            for ad, rec in daily[d].items():
                x = per_ad[ad]
                x['train_clicks'] += rec['clicks']
                x['train_purch'] += rec['purch']
        for d in test_days:
            if d not in daily: continue
            for ad, rec in daily[d].items():
                x = per_ad[ad]
                x['clicks'] += rec['clicks']
                x['spend'] += rec['spend']
                x['purch'] += rec['purch']
                # merge placement mix and carry identifiers
                for mk, mv in (rec.get('placement_mix') or {}).items():
                    if mk not in x['placement_mix']:
                        x['placement_mix'][mk] = {'clicks':0.0,'spend':0.0,'impr':0.0}
                    x['placement_mix'][mk]['clicks'] += mv.get('clicks',0.0)
                    x['placement_mix'][mk]['spend'] += mv.get('spend',0.0)
                    x['placement_mix'][mk]['impr'] += mv.get('impr',0.0)
                x['ad_name'] = x['ad_name'] or rec.get('ad_name')
                x['adset_id'] = x['adset_id'] or rec.get('adset_id')
        # Also accumulate a per-day frequency series over train window for a simple fatigue slope
        day_index = {d:i for i,d in enumerate(all_days)}
        for d in train_days:
            if d not in daily: continue
            for ad, rec in daily[d].items():
                per_ad[ad]['freq_series'].append((day_index[d], float(rec.get('freq') or 0.0)))
        # predicted purchases = E[CVR]*clicks
        items = []
        for ad, agg in per_ad.items():
            clicks = agg['clicks']
            spend = agg['spend']
            purch = agg['purch']
            e_cvr = alpha0 / max(1e-6, (alpha0 + beta0))
            pred_purch = clicks * e_cvr
            sim_score = float(pred_purch)
            actual_cac = (spend/purch) if purch > 0 else float('inf')
            # Normalize placement mix by impressions
            mix = agg.get('placement_mix') or {}
            tot_impr = sum(v.get('impr',0.0) for v in mix.values()) or 1.0
            mix_norm = {k: round(v.get('impr',0.0)/tot_impr,4) for k,v in mix.items()}
            # CPC summary
            cpc = (spend/clicks) if clicks>0 else float('inf')
            # Keep raw placement stats (top 10 by impressions) for richer features
            stats_sorted = sorted(mix.items(), key=lambda kv: kv[1].get('impr',0.0), reverse=True)[:10]
            placement_stats = {k:{'clicks':round(v.get('clicks',0.0),3), 'spend':round(v.get('spend',0.0),3), 'impr':round(v.get('impr',0.0),1)} for k,v in stats_sorted}
            # Fatigue slope over train window using frequency as proxy
            slope = None
            fs = agg.get('freq_series') or []
            if len(fs) >= 3:
                xs = [x for x,_ in fs]; ys=[y for _,y in fs]
                n=len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in fs)
                denom = (n*sxx - sx*sx)
                if denom != 0:
                    slope = (n*sxy - sx*sy)/denom
            items.append({
                'creative_id': ad,
                'adset_id': agg.get('adset_id'),
                'sim_score': sim_score,
                'actual_score': purch,
                'actual_cac': actual_cac,
                'placement_mix': mix_norm,
                'placement_stats': placement_stats,
                'ad_name': agg.get('ad_name'),
                'train_clicks': round(float(agg.get('train_clicks') or 0.0), 3),
                'train_purch': round(float(agg.get('train_purch') or 0.0), 3),
                'test_clicks': round(float(clicks or 0.0), 3),
                'test_spend': round(float(spend or 0.0), 3),
                'test_cpc': round(float(cpc), 6) if math.isfinite(cpc) else None,
                'fatigue_slope': round(float(slope), 6) if slope is not None else None,
            })

        if not items:
            # nothing to write for this campaign
            continue

        # label positives conservatively: require non-trivial volume and be in good CAC/purchases quantiles
        def quantile(vals: List[float], q: float) -> float:
            if not vals: return float('nan')
            s = sorted(vals); k = max(0, min(len(s)-1, int(q*(len(s)-1))))
            return float(s[k])
        purch_vals = [it['actual_score'] for it in items]
        cac_vals = [it['actual_cac'] for it in items if math.isfinite(it['actual_cac'])]
        q_purch = quantile(purch_vals, float(os.getenv('AELP2_PCT_PURCH_POS','0.60')))
        q_cac = quantile(cac_vals, float(os.getenv('AELP2_PCT_CAC_POS','0.40')))
        labeled = []
        pos_count = 0
        for it in items:
            vol_ok = (it['actual_score'] >= MIN_PURCH) or (it['test_clicks'] >= MIN_CLICKS) or (it['test_spend'] >= MIN_SPEND)
            worked = 1 if (vol_ok and it['actual_score'] >= q_purch and it['actual_cac'] <= q_cac) else 0
            pos_count += worked
            labeled.append({
                'creative_id': it['creative_id'],
                'sim_score': round(float(it['sim_score']), 6),
                'actual_label': worked,
                'actual_score': round(float(it['actual_score']), 6),
                'test_clicks': it['test_clicks'],
                'test_spend': it['test_spend'],
                'test_cpc': it['test_cpc'],
                'fatigue_slope': it['fatigue_slope'],
                'placement_mix': it['placement_mix'],
                'placement_stats': it['placement_stats'],
            })
        # Avoid degenerate all-positive or all-negative campaigns by relaxing/tightening thresholds
        if pos_count == 0 and items:
            # Loosen purchases threshold
            q_purch = quantile(purch_vals, 0.50)
            for it in labeled:
                vol_ok = (it['actual_score'] >= MIN_PURCH) or (it['test_clicks'] >= MIN_CLICKS) or (it['test_spend'] >= MIN_SPEND)
                it['actual_label'] = 1 if (vol_ok and it['actual_score'] >= q_purch and (it['test_cpc'] or float('inf')) <= q_cac) else 0
        if pos_count == len(labeled) and labeled:
            # Tighten CAC threshold
            q_cac = quantile(cac_vals, 0.30)
            for it in labeled:
                it['actual_label'] = 1 if (it['actual_score'] >= q_purch and (it['test_cpc'] or float('inf')) <= q_cac) else 0

        out = {'campaign_id': camp_id, 'items': labeled}
        (OUTDIR / f'{camp_id}.json').write_text(json.dumps(out, indent=2))
        print(f'Wrote {OUTDIR}/{camp_id}.json ({len(labeled)} items)')


if __name__ == '__main__':
    main()
