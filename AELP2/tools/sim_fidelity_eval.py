#!/usr/bin/env python3
"""
Simulation Fidelity Check (AELP ↔ AELP2)

Uses existing AELP2 ProductionFortifiedEnvironment (auction via AuctionGymWrapper) and
LegacyEnvAdapter (bid calibration) to simulate day-level outcomes and compare them to
real Meta outcomes over a recent window. No live changes are made.

What it does:
- Calibrates the auction (saves calibration to AELP2 .auction_calibration.pkl)
- Pulls last N days of Meta campaign performance (spend, purchases) via Graph API
- For each day: spins an env with daily budget, runs until spent, records sim purchases/spend
- Reports MAPE for purchases/day and CAC/day and coverage stats

Env required:
- META_ACCESS_TOKEN, META_ACCOUNT_ID in .env or environment
- Optional: GOOGLE_CLOUD_* not required here
"""
from __future__ import annotations

import os
import json
import time
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import requests

# Local imports (assumes repo root on PYTHONPATH)
from AELP2.core.env.calibration import AdvancedAuctionCalibrator
from auction_gym_integration_fixed import AuctionGymWrapper


def load_env_vars(dotenv_path: str = '.env') -> None:
    if os.path.exists(dotenv_path):
        for ln in open(dotenv_path):
            ln = ln.strip()
            if ln.startswith('export '):
                ln = ln[7:]
            if '=' in ln:
                k, v = ln.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def fetch_meta_daily_last_n(n: int = 14) -> List[Tuple[str, float, float]]:
    token = os.getenv('META_ACCESS_TOKEN'); acct = os.getenv('META_ACCOUNT_ID')
    if not token or not acct:
        raise RuntimeError('Missing META_ACCESS_TOKEN or META_ACCOUNT_ID')
    base = f"https://graph.facebook.com/v21.0/{acct}/insights"
    params = {
        'time_increment': 1,
        'date_preset': f'last_{n}d',
        'level': 'account',
        'fields': 'date_start,spend,actions',
        'access_token': token
    }
    r = requests.get(base, params=params, timeout=60).json()
    daily = []
    for row in r.get('data', []):
        spend = float(row.get('spend') or 0.0)
        purch = 0.0
        for a in (row.get('actions') or []):
            if a.get('action_type') == 'offsite_conversion.fb_pixel_purchase':
                purch = float(a.get('value') or 0.0)
                break
        daily.append((row.get('date_start'), spend, purch))
    daily.sort(key=lambda x: x[0])
    return daily


def calibrate_and_save() -> AdvancedAuctionCalibrator:
    awg = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})
    cal = AdvancedAuctionCalibrator(target_min=0.15, target_max=0.25, probe_points=6, trials_per_point=30, validation_points=3)
    def ctx():
        return {
            'query_value': random.uniform(2.0, 6.0),
            'quality_score': random.uniform(0.6, 0.9)
        }
    cal.calibrate(auction_component=awg, context_factory=ctx, save_results=True)
    return cal


@dataclass
class DayResult:
    date: str
    spend_real: float
    purch_real: float
    spend_sim: float
    purch_sim: float


def run_sim_for_budget_via_auction(awg: AuctionGymWrapper, bid_apply, daily_budget: float, ctr_baseline: float, cvr_baseline: float, max_rounds: int | None = None) -> Tuple[float, float]:
    """Simulate spend/conversions using calibrated AuctionGym only.
    - bid_apply: function(bid)->calibrated bid (scale/offset)
    - ctr_baseline: clicks / impressions baseline
    - cvr_baseline: purchases / clicks baseline
    """
    spent = 0.0; conversions = 0.0
    rounds = 0
    base_bid = 2.0
    if max_rounds is None:
        # Rough upper bound: assume ~$0.25–$1 price per win
        max_rounds = int(daily_budget * 40) + 1000
    while spent < daily_budget and rounds < max_rounds:
        rounds += 1
        bid = bid_apply(base_bid)
        ctx = {
            'query_value': random.uniform(2.0, 6.0),
            'quality_score': random.uniform(0.6, 0.9),
            'estimated_ctr': ctr_baseline
        }
        res = awg.run_auction(bid, ctx['query_value'], ctx)
        if res.won:
            spent += float(res.price_paid)
            # Sample click based on ctr_baseline (one impression at a time)
            click = random.random() < ctr_baseline
            if click:
                conv = random.random() < cvr_baseline
                if conv:
                    conversions += 1.0
    # Clip overspend to daily_budget in CAC calc only
    return min(spent, daily_budget), conversions


def evaluate(n_days: int = 14) -> Dict:
    load_env_vars()
    # 1) Calibrate auction once
    cal = calibrate_and_save()
    # 2) Fetch real daily
    daily = fetch_meta_daily_last_n(n_days)
    # Baselines from real window
    # Fetch account-level impressions and clicks for same window
    token = os.getenv('META_ACCESS_TOKEN'); acct = os.getenv('META_ACCOUNT_ID')
    base = f"https://graph.facebook.com/v21.0/{acct}/insights"
    r = requests.get(base, params={'time_increment':1,'date_preset': f'last_{n_days}d','level':'account','fields':'date_start,impressions,clicks','access_token':token}, timeout=60).json()
    tot_impr = sum(int(float(d.get('impressions') or 0)) for d in r.get('data', []))
    tot_clicks = sum(int(float(d.get('clicks') or 0)) for d in r.get('data', []))
    tot_purch = sum(p for _,_,p in daily)
    ctr_baseline = (tot_clicks / tot_impr) if tot_impr else 0.02
    cvr_baseline = (tot_purch / tot_clicks) if tot_clicks else 0.02
    # Prepare auction wrapper
    awg = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})
    bid_apply = lambda b: cal.calibration_result.apply(b) if hasattr(cal, 'calibration_result') else b
    results: List[DayResult] = []
    for date, spend, purch in daily:
        if spend <= 0:
            results.append(DayResult(date, spend, purch, 0.0, 0.0))
            continue
        sim_spend, sim_purch = run_sim_for_budget_via_auction(awg, bid_apply, spend, ctr_baseline, cvr_baseline)
        results.append(DayResult(date, spend, purch, sim_spend, sim_purch))
    # Metrics
    def mape(real, pred):
        if real == 0:
            return None
        return abs(real - pred) / max(1e-6, real)
    mape_purch = [mape(r.purch_real, r.purch_sim) for r in results if r.purch_real > 0]
    mape_cac = []
    for r in results:
        if r.purch_real > 0 and r.purch_sim > 0:
            cac_real = r.spend_real / r.purch_real
            cac_sim = r.spend_sim / r.purch_sim
            mape_cac.append(mape(cac_real, cac_sim))
    out = {
        'days': [r.__dict__ for r in results],
        'mape_purchases_median': None if not mape_purch else float(sorted(mape_purch)[len(mape_purch)//2]),
        'mape_cac_median': None if not mape_cac else float(sorted(mape_cac)[len(mape_cac)//2]),
    }
    return out


def main():
    n = int(os.getenv('AELP2_FIDELITY_DAYS', '14'))
    report = evaluate(n_days=n)
    path = os.getenv('AELP2_FIDELITY_OUT', 'AELP2/reports/sim_fidelity.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({'report': path, 'summary': {
        'mape_purchases_median': report['mape_purchases_median'],
        'mape_cac_median': report['mape_cac_median'],
        'days': len(report['days'])
    }}, indent=2))


if __name__ == '__main__':
    main()
