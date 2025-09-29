#!/usr/bin/env python3
import os, json, random, requests
from dataclasses import dataclass
from typing import List

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


def load_env():
    p = os.path.expanduser('~/AELP/.env')
    if os.path.exists(p):
        for ln in open(p):
            if ln.startswith('export '):
                ln = ln[7:]
            if '=' in ln:
                k, v = ln.strip().split('=', 1)
                os.environ.setdefault(k, v)


def fetch_daily(n: int = 14) -> List[Row]:
    token = os.environ['META_ACCESS_TOKEN']
    acct = os.environ['META_ACCOUNT_ID']
    base = f"https://graph.facebook.com/v21.0/{acct}/insights"
    params = {
        'time_increment': 1,
        'date_preset': f'last_{n}d',
        'level': 'account',
        'fields': 'date_start,spend,impressions,clicks,actions',
        'access_token': token,
    }
    r = requests.get(base, params=params, timeout=60).json()
    out: List[Row] = []
    for d in r.get('data', []):
        purch = 0.0
        for a in (d.get('actions') or []):
            if a.get('action_type') == 'offsite_conversion.fb_pixel_purchase':
                purch = float(a.get('value') or 0.0)
                break
        out.append(Row(
            d.get('date_start'),
            float(d.get('spend') or 0.0),
            int(float(d.get('impressions') or 0)),
            int(float(d.get('clicks') or 0)),
            purch,
        ))
    out.sort(key=lambda x: x.date)
    return out


def simulate_empirical(rows: List[Row]):
    m = len(rows)
    k = max(7, m // 2)
    train, test = rows[:k], rows[k:]
    tot_clicks = sum(r.clicks for r in train) or 1
    tot_impr = sum(r.impr for r in train) or 1
    tot_spend = sum(r.spend for r in train) or 1.0
    tot_purch = sum(r.purch for r in train)
    cpc_samples = [r.spend / r.clicks for r in train if r.clicks > 0]
    if not cpc_samples:
        cpc_samples = [tot_spend / tot_clicks]
    ctr_train = tot_clicks / tot_impr
    cvr_train = (tot_purch / tot_clicks) if tot_clicks > 0 else 0.01
    results: List[DayResult] = []
    for r in test:
        budget = r.spend
        spent = 0.0
        clicks = 0
        avg_cpc = sum(cpc_samples) / len(cpc_samples)
        max_iter = int(budget / max(0.01, avg_cpc)) + 200
        for _ in range(max_iter):
            cpc = random.choice(cpc_samples)
            if spent + cpc > budget:
                break
            spent += cpc
            clicks += 1
        purch = 0
        for _ in range(clicks):
            if random.random() < cvr_train:
                purch += 1
        results.append(DayResult(r.date, r.spend, r.purch, spent, float(purch)))
    def mape(a, b):
        return None if a <= 0 else abs(a - b) / a
    mape_p = [mape(r.purch_real, r.purch_sim) for r in results if r.purch_real > 0]
    mape_c = []
    for r in results:
        if r.purch_real > 0 and r.purch_sim > 0:
            mape_c.append(mape(r.spend_real / r.purch_real, r.spend_sim / max(1.0, r.purch_sim)))
    rep = {
        'train_days': k,
        'test_days': len(test),
        'ctr_train': ctr_train,
        'cvr_train': cvr_train,
        'cpc_mean_train': sum(cpc_samples) / len(cpc_samples),
        'mape_purchases_median': None if not mape_p else sorted(mape_p)[len(mape_p) // 2],
        'mape_cac_median': None if not mape_c else sorted(mape_c)[len(mape_c) // 2],
        'days': [r.__dict__ for r in results],
    }
    return rep


if __name__ == '__main__':
    load_env()
    rows = fetch_daily(14)
    out = simulate_empirical(rows)
    path = 'AELP2/reports/sim_fidelity_empirical.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({'report': path, 'summary': {k: out[k] for k in ('train_days', 'test_days', 'mape_purchases_median', 'mape_cac_median')}}, indent=2))

