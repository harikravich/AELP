#!/usr/bin/env python3
"""
Head-to-head comparison between live and new slates; reports uplift with CIs.

Usage:
  python pipelines/eval/head_to_head.py --live data/live.csv --new data/new.csv --out artifacts/uplift_report.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def rate_and_ci(successes, trials, alpha=0.05):
    # Wilson score interval
    from math import sqrt
    if trials == 0:
        return 0.0, (0.0, 0.0)
    p = successes / trials
    z = 1.96  # approx for 95%
    denom = 1 + z*z/trials
    centre = p + z*z/(2*trials)
    adj = z*sqrt((p*(1-p)+z*z/(4*trials))/trials)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return p, (max(0.0, lo), min(1.0, hi))


def compare(live: pd.DataFrame, new: pd.DataFrame):
    Lc, Li = live['conversions'].sum(), live['impressions'].sum()
    Nc, Ni = new['conversions'].sum(), new['impressions'].sum()
    pr, ci_r = rate_and_ci(Lc, Li)
    pn, ci_n = rate_and_ci(Nc, Ni)
    uplift = pn - pr
    return {
        'live': {'cr': pr, 'ci': ci_r, 'n': int(Li)},
        'new': {'cr': pn, 'ci': ci_n, 'n': int(Ni)},
        'uplift': uplift,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    live = pd.read_csv(args.live)
    new = pd.read_csv(args.new)
    rep = compare(live, new)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"Uplift: {rep['uplift']:.4f} (live CR={rep['live']['cr']:.4f} new CR={rep['new']['cr']:.4f})")


if __name__ == '__main__':
    main()

