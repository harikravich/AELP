#!/usr/bin/env python3
from __future__ import annotations
"""
Offline bandit simulation using the forecast distributions for Top-20 blueprints.

- Thompson Sampling with Beta-Bernoulli proxy on "success" = signups rate per impression,
  calibrated from forecast p50 CTR/CVR and CPM for each blueprint.
- Simulates allocation over T days for a chosen daily budget.
Outputs: AELP2/reports/rl_offline_simulation.json
"""
import json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORE = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
OUT = ROOT / 'AELP2' / 'reports' / 'rl_offline_simulation.json'


def pick_metric(bud: dict):
    # Use signups per impression proxy and CAC p50 to drive priors
    imps = float(bud['impressions']['p50'])
    su = float(bud['signups']['p50'])
    cac = float(bud['cac']['p50'])
    rate = su / max(imps, 1.0)
    return rate, cac


def run(days=3, daily_budget=30000.0):
    f = json.loads(FORE.read_text())
    items = f['items']
    arms = []
    for r in items:
        cid = r['creative_id']
        b = r['budgets'][str(int(daily_budget))]
        rate, cac = pick_metric(b)
        # Beta prior k set by pseudo-impressions; scale to get reasonable sharpness
        k = 2000.0
        a = max(1.0, rate * k)
        bpar = max(1.0, (1.0 - rate) * k)
        arms.append({'id': cid, 'a': a, 'b': bpar, 'spend': 0.0, 'wins': 0.0, 'imps': 0.0, 'cac_p50': cac})

    history = []
    for t in range(days):
        # sample success rates from priors
        samples = [(i, random.betavariate(arm['a'], arm['b'])) for i, arm in enumerate(arms)]
        samples.sort(key=lambda x: x[1], reverse=True)
        # allocate top quartile equally
        q = max(1, len(arms)//4)
        winners = [i for i, _ in samples[:q]]
        alloc = daily_budget / q
        for i, arm in enumerate(arms):
            spend = alloc if i in winners else 0.0
            # outcome simulation: convert spend -> imps -> signups using sampled rate
            # approximate CPM from signups and cac_p50 at p50: spend ≈ CAC * signups ⇒ signups_p50 ≈ spend / CAC
            su_p50 = max(1.0, spend / max(arm['cac_p50'], 1e-6))
            # derive imps_p50 from rate: imps ≈ signups / rate
            imps = su_p50 / max((arm['a'] / (arm['a'] + arm['b'])), 1e-6)
            # sample noise around rate
            r_samp = random.betavariate(arm['a'], arm['b'])
            su_obs = r_samp * imps
            # update priors
            arm['a'] += su_obs
            arm['b'] += max(imps - su_obs, 0.0)
            arm['spend'] += spend
            arm['wins'] += su_obs
            arm['imps'] += imps
        history.append({'day': t+1, 'allocations': [{'id': arms[i]['id'], 'spend': alloc if i in winners else 0.0} for i in range(len(arms))]})

    arms.sort(key=lambda a: a['wins']/max(a['spend'],1.0), reverse=True)
    OUT.write_text(json.dumps({'days': days, 'daily_budget': daily_budget, 'ranking': arms[:10], 'history': history}, indent=2))
    print(json.dumps({'out': str(OUT), 'top': [a['id'] for a in arms[:5]]}, indent=2))


if __name__ == '__main__':
    run()

