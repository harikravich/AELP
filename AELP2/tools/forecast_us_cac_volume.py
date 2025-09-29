#!/usr/bin/env python3
from __future__ import annotations
"""
Monte Carlo CAC/volume forecast for Top-20 blueprints at $30k and $50k budgets.

Inputs:
  - us_meta_baselines.json (CPM/CTR/CVR p10/p50/p90)
  - ad_blueprints_top20.json (p_win, lcb)

Outputs:
  - AELP2/reports/us_cac_volume_forecasts.json

Assumptions:
  - CTR multiplier f(p) = 0.5 + 1.0 * p  (0.5x..1.5x)
  - CVR multiplier g(p) = 0.8 + 0.4 * p  (0.8x..1.2x)
  - CPM, CTR, CVR draws from triangular(p10,p50,p90); CTR expressed as rate (0..1).
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'AELP2' / 'reports' / 'us_meta_baselines.json'
TOP = ROOT / 'AELP2' / 'reports' / 'ad_blueprints_top20.json'
OUT = ROOT / 'AELP2' / 'reports' / 'us_cac_volume_forecasts.json'


def tri(p10, p50, p90):
    # simple triangular distribution: min=p10, mode=p50, max=p90
    return random.triangular(p10, p90, p50)


def run(n=5000):
    base = json.loads(BASE.read_text())
    top = json.loads(TOP.read_text())['items']

    # Convert baselines
    cpm_p10 = float(base['cpm_p10']); cpm_p50 = float(base['cpm_p50']); cpm_p90 = float(base['cpm_p90'])
    ctr_p10 = float(base['ctr_p10']); ctr_p50 = float(base['ctr_p50']); ctr_p90 = float(base['ctr_p90'])
    # CVR/CTR percentiles from baseline (already aggregated daily) — use these, not totals
    cvr_p10 = float(base.get('cvr_p10', 0.01))
    cvr_p50 = float(base.get('cvr_p50', 0.02))
    cvr_p90 = float(base.get('cvr_p90', 0.04))
    # Guard: CTR/CVR are rates; ensure in [0,1], and clamp CVR to a realistic paid range
    def clamp01(x):
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.02
    ctr_p10, ctr_p50, ctr_p90 = map(clamp01, (ctr_p10, ctr_p50, ctr_p90))
    cvr_p10, cvr_p50, cvr_p90 = map(clamp01, (cvr_p10, cvr_p50, cvr_p90))
    # Paid CVR clamp: 0.2%..5%
    def clamp_cvr(x):
        return max(0.002, min(0.05, float(x)))
    cvr_p10, cvr_p50, cvr_p90 = map(clamp_cvr, (cvr_p10, cvr_p50, cvr_p90))
    # Ensure ordering
    if not (cvr_p10 <= cvr_p50 <= cvr_p90):
        cvr_p10 = min(cvr_p10, cvr_p50, cvr_p90)
        cvr_p90 = max(cvr_p10, cvr_p50, cvr_p90)
        cvr_p50 = max(cvr_p10, min(cvr_p50, cvr_p90))

    budgets = [30000.0, 50000.0]
    out = {'budgets': budgets, 'items': []}
    for r in top:
        p = float(r['p_win']); lcb = float(r.get('lcb', 0.0))
        # multipliers from p_win and conservative lcb
        def mult_ctr(pp):
            return 0.5 + 1.0*pp
        def mult_cvr(pp):
            return 0.8 + 0.4*pp
        row = {'creative_id': r['creative_id'], 'p_win': p, 'lcb': lcb, 'budgets': {}}
        for B in budgets:
            sims = []
            for _ in range(n):
                cpm = tri(cpm_p10, cpm_p50, cpm_p90)  # $ per 1000 imps
                ctr = tri(ctr_p10, ctr_p50, ctr_p90) * mult_ctr(p)
                cvr = tri(cvr_p10, cvr_p50, cvr_p90) * mult_cvr(p)
                imps = (B / cpm) * 1000.0
                clicks = imps * ctr
                signups = clicks * cvr
                cac = B / max(signups, 1e-6)
                sims.append((imps, clicks, signups, cac))
            sims.sort(key=lambda t: t[3])  # sort by CAC
            imps_list, clicks_list, su_list, cac_list = zip(*sims)
            def pct(a, q):
                i = int(q*(len(a)-1)); return float(a[i])
            # Probability CAC <= 240 (anchor)
            prob_cac_le_240 = sum(1 for c in cac_list if c <= 240.0) / len(cac_list)
            prob_cac_le_200 = sum(1 for c in cac_list if c <= 200.0) / len(cac_list)
            row['budgets'][str(int(B))] = {
                'impressions': {'p10': pct(imps_list, 0.1), 'p50': pct(imps_list, 0.5), 'p90': pct(imps_list, 0.9)},
                'clicks': {'p10': pct(clicks_list, 0.1), 'p50': pct(clicks_list, 0.5), 'p90': pct(clicks_list, 0.9)},
                'signups': {'p10': pct(su_list, 0.1), 'p50': pct(su_list, 0.5), 'p90': pct(su_list, 0.9)},
                'cac': {'p10': pct(cac_list, 0.1), 'p50': pct(cac_list, 0.5), 'p90': pct(cac_list, 0.9)},
                'p_cac_le_240': round(prob_cac_le_240, 4),
                'p_cac_le_200': round(prob_cac_le_200, 4),
            }
        out['items'].append(row)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({'out': str(OUT), 'count': len(out['items'])}, indent=2))


if __name__ == '__main__':
    run()
