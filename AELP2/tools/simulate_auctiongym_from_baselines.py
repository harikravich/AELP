#!/usr/bin/env python3
from __future__ import annotations
"""
Auction-aware offline simulation (no RecSim) calibrated from current forecasts.

What it does
- Reads placement-aware forecasts: AELP2/reports/us_cac_volume_forecasts.json
- Approximates CTR and CVR from p50 clicks/impressions and signups/clicks per creative.
- Uses AuctionGymWrapper to simulate second-price auctions with aggressive competitors.
- Estimates win rate, CPC, CPM, and CAC under competition pressure for each creative.
- Writes: AELP2/reports/auctiongym_offline_simulation.json

This provides deeper insights than pure Thompson Sampling by introducing
explicit auction dynamics and bid shading. RecSim is not required.
"""
import json, math, random
from statistics import mean
from pathlib import Path
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[2]
FORE = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
OUT = ROOT / 'AELP2' / 'reports' / 'auctiongym_offline_simulation.json'

# Local import of AuctionGym wrapper (pure python)
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


def simulate_for_creative(auction: AuctionGymWrapper, cid: str, bud: Dict[str, Any], aov: float = 120.0, n_rounds: int = 2000) -> Dict[str, Any]:
    rates = derive_rates(bud)
    ctr, cvr = rates['ctr'], rates['cvr']
    # Value of an impression ≈ AOV * P(conversion | impression)
    v_imp = aov * (ctr * cvr)
    # Quality score proxies creative relevance; map CTR into [0.5, 1.5]
    quality = 0.5 + 2.0 * min(0.5, ctr / 0.05)
    # Bid relative to AuctionGym's query_value scale (quality*10), then cap below competitor max ($6)
    query_value = quality * 10.0
    our_bid = min(5.8, max(0.2, 0.4 * query_value))

    wins = 0
    spend = 0.0
    paid_clicks = 0
    imps = 0
    # Store winning price for CPC estimation when outcome clicked
    prices = []

    for _ in range(n_rounds):
        res = auction.run_auction(our_bid, quality, context={'estimated_ctr': ctr})
        if res.won:
            imps += 1
            wins += 1
            # Assume a click occurs with prob=ctr when we win an impression
            if random.random() < ctr:
                paid_clicks += 1
                spend += res.price_paid
                prices.append(res.price_paid)

    win_rate = wins / max(n_rounds, 1)
    cpc = spend / max(paid_clicks, 1)
    cpm = (spend / max(imps, 1)) * 1000.0
    # CAC ≈ CPC / CVR (under p50 cvr assumption); if no clicks, set None
    cac = (cpc / max(cvr, 1e-6)) if paid_clicks else None

    return {
        'creative_id': cid,
        'win_rate': round(win_rate, 4),
        'est_ctr': round(ctr, 5),
        'est_cvr': round(cvr, 5),
        'cpc': round(cpc, 4),
        'cpm': round(cpm, 2),
        'cac': round(cac, 2) if cac is not None else None,
        'our_bid': round(our_bid, 4),
        'quality': round(quality, 3),
        'clicks_sim': paid_clicks,
        'imps_sim': imps,
        'avg_win_price': round(mean(prices), 4) if prices else 0.0,
    }


def main():
    data = json.loads(FORE.read_text())
    items = data.get('items', [])
    # Default budget key used in forecasts
    budget_key = next(iter(items[0]['budgets'].keys())) if items else '30000'
    auction = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})

    # Simulate for top 12 creatives by p_win if available
    items_sorted = sorted(items, key=lambda r: float(r.get('p_win', 0.0)), reverse=True)[:12]

    results = []
    for r in items_sorted:
        cid = r['creative_id']
        bud = r['budgets'][budget_key]
        # Use Security AOV 200 if security creative id pattern, else Balance 120
        aov = 200.0 if cid.startswith('bp_') else 120.0
        sim = simulate_for_creative(auction, cid, bud, aov=aov, n_rounds=2000)
        # Also capture forecast p50 CAC for reference
        sim['forecast_cac_p50'] = round(float(bud['cac']['p50']), 2)
        results.append(sim)

    # Rank by CAC improvement vs forecast (lower is better)
    for s in results:
        s['delta_cac_vs_forecast'] = round(s['cac'] - s['forecast_cac_p50'], 2)

    results.sort(key=lambda x: x['cac'])
    OUT.write_text(json.dumps({'budget_key': budget_key, 'items': results}, indent=2))
    print(json.dumps({'out': str(OUT), 'top5': [r['creative_id'] for r in results[:5]]}, indent=2))


if __name__ == '__main__':
    main()
