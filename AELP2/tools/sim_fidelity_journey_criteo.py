#!/usr/bin/env python3
"""
Journey + Criteo fidelity evaluation (no BigQuery writes).

Train→test split (time-based):
- Train on first K days (estimate CPC distribution and base CVR)
- Test on next days: for each day, simulate spend → clicks using CPC samples,
  then per-click conversion probability adjusted by Criteo CTR vs train CTR.

Outputs JSON with purchases/day MAPE and CAC/day MAPE on held-out days.
"""
from __future__ import annotations

import os, json, random, math, requests
from dataclasses import dataclass
from typing import List

# Ensure repo root on PYTHONPATH
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from criteo_response_model import CriteoUserResponseModel
import numpy as np
from time import time


@dataclass
class Row:
    date: str
    spend: float
    impr: int
    clicks: int
    purch: float


@dataclass
class DayResult:
    date: str
    spend_real: float
    purch_real: float
    spend_sim: float
    purch_sim: float


def load_env(env_path: str) -> None:
    if os.path.exists(env_path):
        for ln in open(env_path):
            if ln.startswith('export '): ln = ln[7:]
            if '=' in ln:
                k, v = ln.strip().split('=', 1)
                os.environ.setdefault(k, v)


def fetch_daily(n: int = 14) -> List[Row]:
    token = os.environ['META_ACCESS_TOKEN']; acct = os.environ['META_ACCOUNT_ID']
    base = f"https://graph.facebook.com/v21.0/{acct}/insights"
    params = {'time_increment': 1, 'date_preset': f'last_{n}d', 'level': 'account',
              'fields': 'date_start,spend,impressions,clicks,actions', 'access_token': token}
    r = requests.get(base, params=params, timeout=60).json()
    out: List[Row] = []
    for d in r.get('data', []):
        purch = 0.0
        for a in (d.get('actions') or []):
            if a.get('action_type') == 'offsite_conversion.fb_pixel_purchase':
                purch = float(a.get('value') or 0.0)
                break
        out.append(Row(
            date=d.get('date_start'),
            spend=float(d.get('spend') or 0.0),
            impr=int(float(d.get('impressions') or 0)),
            clicks=int(float(d.get('clicks') or 0)),
            purch=purch,
        ))
    out.sort(key=lambda x: x.date)
    return out


def simulate_journey_criteo(rows: List[Row]) -> dict:
    m = len(rows)
    k = max(7, m // 2)
    train, test = rows[:k], rows[k:]

    # Empirical training stats
    tot_clicks = sum(r.clicks for r in train) or 1
    tot_impr = sum(r.impr for r in train) or 1
    tot_spend = sum(r.spend for r in train) or 1.0
    tot_purch = sum(r.purch for r in train)
    # CPC distribution with guardrails
    raw_cpcs = [r.spend / r.clicks for r in train if r.clicks > 0]
    if not raw_cpcs:
        raw_cpcs = [tot_spend / tot_clicks]
    p5 = np.percentile(raw_cpcs, 5) if len(raw_cpcs) > 3 else min(raw_cpcs)
    p95 = np.percentile(raw_cpcs, 95) if len(raw_cpcs) > 3 else max(raw_cpcs)
    cpc_floor = max(0.05, float(p5))
    cpc_cap = max(cpc_floor, float(p95))
    cpc_samples = [float(np.clip(c, cpc_floor, cpc_cap)) for c in raw_cpcs]
    ctr_train = tot_clicks / tot_impr
    cvr_train = (tot_purch / tot_clicks) if tot_clicks > 0 else 0.01

    # Load Criteo CTR model
    criteo = CriteoUserResponseModel()

    # Build a small base CTR set (few model calls) and sample with lognormal noise
    base_ctrs = []
    for dev in ['mobile', 'desktop']:
        for sg in ['researcher', 'concerned_parent', 'price_sensitive']:
            ad = {'price': 99, 'category': 'security', 'format': 'video'}
            ctx = {'user_segment': sg, 'device': dev, 'session_duration': 8, 'page_views': 4, 'geo_region': 'US'}
            out = criteo.simulate_user_response('seed', ad, ctx)
            base_ctrs.append(float(out.get('predicted_ctr', 0.05)))
    base_ctrs = np.clip(np.array(base_ctrs, dtype=float), 1e-4, 0.5)
    rng = np.random.default_rng(123)
    def sample_ctr_fast(size: int) -> np.ndarray:
        base = rng.choice(base_ctrs, size=size)
        noise = rng.lognormal(mean=0.0, sigma=0.25, size=size)
        return np.clip(base * noise, 1e-4, 0.9)

    # Calibrate gamma and weight w on training window so predicted purchases match observed
    # conv_p(click) = cvr_train * ((1-w) + w*(ctr/ctr_train)**gamma)
    # Optional calibration with strict caps; default OFF
    def simulate_total_purchases(rows_subset: List[Row], gamma: float, w: float) -> float:
        total = 0.0
        for r in rows_subset:
            budget = r.spend
            min_cpc = max(cpc_floor, 0.05)
            max_clicks_cap = int(min(20000, budget / min_cpc + 500))
            draw = rng.choice(cpc_samples, size=max_clicks_cap)
            csum = np.cumsum(draw)
            clicks = int(np.searchsorted(csum, budget, side='right'))
            clicks = max(0, min(clicks, max_clicks_cap))
            if clicks <= 0:
                continue
            ctrs = sample_ctr_fast(clicks)
            scale = (ctrs / max(1e-4, ctr_train)) ** gamma
            conv_p = cvr_train * ((1.0 - w) + w * scale)
            conv_p = np.clip(conv_p, 0.0005, 0.5)
            rand = rng.random(clicks)
            total += float(np.sum(rand < conv_p))
        return total

    do_calib = os.getenv('AELP2_JC_CALIBRATE', '0') == '1'
    gamma_opt, w_opt = 0.9, 0.5
    if do_calib:
        best = {'err': float('inf'), 'gamma': gamma_opt, 'w': w_opt}
        start_c = time()
        for gamma in [0.6, 0.9, 1.2]:
            for w in [0.3, 0.5, 0.7]:
                if time() - start_c > 10.0:  # 10s cap
                    break
                pred = simulate_total_purchases(train, gamma, w)
                err = abs(pred - tot_purch)
                if err < best['err']:
                    best = {'err': err, 'gamma': gamma, 'w': w}
        gamma_opt, w_opt = best['gamma'], best['w']

    results: List[DayResult] = []
    for r in test:
        budget = r.spend
        # Vectorized click count via CPC cumulative sum
        min_cpc = max(cpc_floor, 0.05)
        max_clicks_cap = int(min(50000, budget / min_cpc + 1000))
        draw = rng.choice(cpc_samples, size=max_clicks_cap)
        csum = np.cumsum(draw)
        clicks = int(np.searchsorted(csum, budget, side='right'))
        clicks = max(0, min(clicks, max_clicks_cap))
        spent = float(csum[clicks-1]) if clicks > 0 else 0.0

        purch = 0
        if clicks > 0:
            ctrs = sample_ctr_fast(clicks)
            scale = (ctrs / max(1e-4, ctr_train)) ** gamma_opt
            conv_p = cvr_train * ((1.0 - w_opt) + w_opt * scale)
            conv_p = np.clip(conv_p, 0.0005, 0.5)
            rand = rng.random(clicks)
            purch = int(np.sum(rand < conv_p))

        results.append(DayResult(r.date, r.spend, r.purch, spent, float(purch)))

    # Metrics
    def mape(a, b):
        return None if a <= 0 else abs(a - b) / a

    mape_p = [mape(r.purch_real, r.purch_sim) for r in results if r.purch_real > 0]
    mape_c = []
    for r in results:
        if r.purch_real > 0 and r.purch_sim > 0:
            mape_c.append(mape(r.spend_real / r.purch_real, r.spend_sim / max(1.0, r.purch_sim)))

    report = {
        'train_days': k,
        'test_days': len(test),
        'ctr_train': ctr_train,
        'cvr_train': cvr_train,
        'cpc_mean_train': sum(cpc_samples) / len(cpc_samples),
        'mape_purchases_median': None if not mape_p else sorted(mape_p)[len(mape_p) // 2],
        'mape_cac_median': None if not mape_c else sorted(mape_c)[len(mape_c) // 2],
        'gamma': gamma_opt,
        'w': w_opt,
        'days': [r.__dict__ for r in results],
    }
    return report


if __name__ == '__main__':
    envp = os.path.expanduser('~/AELP/.env')
    load_env(envp)
    rows = fetch_daily(14)
    out = simulate_journey_criteo(rows)
    path = 'AELP2/reports/sim_fidelity_journey_criteo.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'report': path, 'summary': {k: out[k] for k in ('train_days', 'test_days', 'mape_purchases_median', 'mape_cac_median')}}, indent=2))
