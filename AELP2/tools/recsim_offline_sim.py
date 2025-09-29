#!/usr/bin/env python3
from __future__ import annotations
"""
RecSim-based offline slate simulation (lightweight).

Approach (pragmatic):
- Use RecSimUserModel personas to estimate CTR & CVR multipliers by persona.
- Weight personas with a simple prior (Impulse 0.25, Researcher 0.2, Loyal 0.2, Price 0.15, Window 0.1, Brand 0.1).
- Combine with baseline placement forecasts (us_cac_volume_forecasts.json) per creative.
- Output expected CAC and signups per creative and a ranked slate.

Outputs: AELP2/reports/recsim_offline_simulation.json
"""
import json, random
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
FORE = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'
OUT = ROOT / 'AELP2' / 'reports' / 'recsim_offline_simulation.json'

import sys
sys.path.insert(0, str(ROOT))
from recsim_user_model import RecSimUserModel, UserSegment


PERSONA_WEIGHTS = {
    UserSegment.IMPULSE_BUYER: 0.25,
    UserSegment.RESEARCHER: 0.20,
    UserSegment.LOYAL_CUSTOMER: 0.20,
    UserSegment.PRICE_CONSCIOUS: 0.15,
    UserSegment.WINDOW_SHOPPER: 0.10,
    UserSegment.BRAND_LOYALIST: 0.10,
}


def persona_ctr_cvr(model: RecSimUserModel, n=200) -> Dict[UserSegment, Dict[str,float]]:
    """Empirical CTR/CVR per persona under a generic creative with good quality."""
    out = {}
    for seg in PERSONA_WEIGHTS.keys():
        uid = f"sim_{seg.value}"
        model.generate_user(uid, seg)
        clicks = conv = imps = 0
        for _ in range(n):
            resp = model.simulate_ad_response(uid, {
                'creative_quality': 0.9,
                'price_shown': 50.0,
                'brand_match': 0.7,
                'relevance_score': 0.8
            }, {'hour': 14, 'device': 'mobile'})
            imps += 1
            if resp.get('clicked'):
                clicks += 1
                if resp.get('converted'):
                    conv += 1
        ctr = clicks / max(imps,1)
        cvr = conv / max(clicks,1) if clicks>0 else 0.0
        out[seg] = {'ctr': ctr, 'cvr': cvr}
    return out


def main():
    data = json.loads(FORE.read_text())
    budget_key = next(iter(data['items'][0]['budgets'].keys())) if data.get('items') else '30000'
    model = RecSimUserModel()
    per = persona_ctr_cvr(model, n=300)

    items = []
    for r in data['items']:
        cid = r['creative_id']
        bud = r['budgets'][budget_key]
        # Baseline p50 CTR/CVR and signups
        imps = float(bud['impressions']['p50'])
        clicks = float(bud['clicks']['p50'])
        signups = float(bud['signups']['p50'])
        ctr_base = clicks / max(imps,1.0)
        cvr_base = signups / max(clicks,1.0)
        # Persona-mixed multipliers: relative to base CTR/CVR levels
        ctr_mix = sum(PERSONA_WEIGHTS[s]*per[s]['ctr'] for s in PERSONA_WEIGHTS)
        cvr_mix = sum(PERSONA_WEIGHTS[s]*per[s]['cvr'] for s in PERSONA_WEIGHTS)
        # Avoid zero; scale relative to a nominal baseline (0.02 ctr, 0.02 cvr)
        ctr_mult = (ctr_mix / 0.02) if 0.02>0 else 1.0
        cvr_mult = (cvr_mix / 0.02) if 0.02>0 else 1.0
        ctr_adj = max(1e-6, ctr_base * ctr_mult)
        cvr_adj = max(1e-6, cvr_base * cvr_mult)
        clicks_adj = imps * ctr_adj
        signups_adj = clicks_adj * cvr_adj
        # CAC recompute with same budget
        B = 30000.0
        cac_adj = B / max(signups_adj, 1e-6)
        items.append({'creative_id': cid, 'ctr_adj': ctr_adj, 'cvr_adj': cvr_adj, 'cac_adj': cac_adj, 'signups_adj': signups_adj})

    items.sort(key=lambda x: x['cac_adj'])
    slate = [x['creative_id'] for x in items[:8]]
    OUT.write_text(json.dumps({'budget_key': budget_key, 'slate': slate, 'items': items}, indent=2))
    print(json.dumps({'out': str(OUT), 'top5': slate[:5]}, indent=2))


if __name__ == '__main__':
    main()

