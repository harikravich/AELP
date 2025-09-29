#!/usr/bin/env python3
from __future__ import annotations
"""
AuctionGym sensitivity sweeps over bid and quality multipliers.

Reads placement-aware forecasts: AELP2/reports/us_cac_volume_forecasts.json
For the top creatives, simulates auctions under a grid of multipliers and
reports CAC deltas and rank stability.

Outputs: AELP2/reports/auctiongym_sensitivity.json
"""
import json, random
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
FORE = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
OUT = ROOT / 'AELP2' / 'reports' / 'auctiongym_sensitivity.json'

import sys
sys.path.insert(0, str(ROOT))
from auction_gym_integration import AuctionGymWrapper


def derive_rates(bud: Dict[str, Any]) -> Dict[str, float]:
    imps = float(bud['impressions']['p50']) or 1.0
    clicks = float(bud['clicks']['p50']) or 1.0
    signups = float(bud['signups']['p50']) or 1.0
    ctr = max(1e-6, clicks / imps)
    cvr = max(1e-6, signups / clicks)
    return {'ctr': ctr, 'cvr': cvr}


def simulate(auction: AuctionGymWrapper, ctr: float, cvr: float, bid_mult: float, q_mult: float, rounds: int = 1000) -> Tuple[float,float,float]:
    # Map ctr to quality proxy [0.5..1.5]
    base_quality = 0.5 + 2.0 * min(0.5, ctr / 0.05)
    quality = max(0.3, min(2.0, base_quality * q_mult))
    query_value = quality * 10.0
    # Bid relative to query value (cap below competitor max)
    our_bid = min(5.8, max(0.2, 0.4 * query_value * bid_mult))

    wins = clicks = 0
    spend = 0.0
    imps = 0
    for _ in range(rounds):
        res = auction.run_auction(our_bid, quality, context={'estimated_ctr': ctr})
        if res.won:
            imps += 1
            if random.random() < ctr:
                clicks += 1
                spend += res.price_paid
    cpc = spend / max(clicks, 1)
    cpm = (spend / max(imps, 1)) * 1000.0
    cac = (cpc / max(cvr, 1e-6)) if clicks else None
    return cpc, cpm, (cac if cac is not None else float('inf'))


def main():
    data = json.loads(FORE.read_text())
    # choose first budget present
    budget_key = next(iter(data['items'][0]['budgets'].keys())) if data.get('items') else '30000'
    items = sorted(data['items'], key=lambda r: float(r.get('p_win', 0.0)), reverse=True)[:12]
    auction = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})

    grid = [(b, q) for b in (0.8, 1.0, 1.2) for q in (0.9, 1.0, 1.1)]
    results = {}
    for r in items:
        cid = r['creative_id']
        bud = r['budgets'][budget_key]
        rates = derive_rates(bud)
        rows = []
        for (bm, qm) in grid:
            cpc, cpm, cac = simulate(auction, rates['ctr'], rates['cvr'], bm, qm, rounds=1000)
            rows.append({'bid_mult': bm, 'quality_mult': qm, 'cpc': round(cpc,4), 'cpm': round(cpm,2), 'cac': (round(cac,2) if cac!=float('inf') else None)})
        results[cid] = rows

    # Rank stability: use median CAC across grid
    med = []
    for cid, rows in results.items():
        cacs = [x['cac'] for x in rows if x['cac'] is not None]
        m = sorted(cacs)[len(cacs)//2] if cacs else float('inf')
        med.append((cid, m))
    med_sorted = [cid for cid, _ in sorted(med, key=lambda t: t[1])]

    OUT.write_text(json.dumps({'budget_key': budget_key, 'grid': grid, 'per_creative': results, 'rank_median_cac': med_sorted[:12]}, indent=2))
    print(json.dumps({'out': str(OUT), 'stable_top5': med_sorted[:5]}, indent=2))


if __name__ == '__main__':
    main()

