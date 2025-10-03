#!/usr/bin/env python3
"""
Generate synthetic Meta and GA CSVs for training or holdout.

Usage:
  python pipelines/data/generate_synthetic.py \
    --start 2025-07-01 --end 2025-08-15 --rows-per-day 120 \
    --out-meta artifacts/synth/train/meta.csv \
    --out-ga artifacts/synth/train/ga.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


CAMPAIGNS = ['c1','c2','c3']
ADSETS = ['s1','s2','s3','s4']
ADS = ['a1','a2','a3','a4','a5']
DEVICES = ['mobile','desktop']


def gen_day(date: str, n: int, rng: np.random.Generator):
    rows_meta = []
    rows_ga = []
    for _ in range(n):
        campaign_id = rng.choice(CAMPAIGNS)
        adset_id = rng.choice(ADSETS)
        ad_id = rng.choice(ADS)
        device = rng.choice(DEVICES, p=[0.7, 0.3])
        # Base rates
        base_ctr = 0.02 if device == 'desktop' else 0.035
        base_c2s = 0.04 if campaign_id == 'c1' else (0.025 if device=='mobile' else 0.02)
        base_tps = 0.05
        imps = int(rng.integers(800, 6000))
        clicks = int(rng.binomial(imps, min(0.2, max(0.001, base_ctr + rng.normal(0, 0.005)))))
        spend = float(max(0.5, clicks * (0.2 + abs(rng.normal(0.3, 0.1)))))
        sessions = int(max(clicks, clicks + rng.integers(-5, 30)))
        subs = int(rng.binomial(max(clicks,1), min(0.5, max(0.001, base_c2s + rng.normal(0, 0.01)))))
        tx = int(rng.binomial(max(sessions,1), min(0.6, max(0.001, base_tps + rng.normal(0, 0.01)))))
        rows_meta.append({
            'date': date,
            'campaign_id': campaign_id,
            'adset_id': adset_id,
            'ad_id': ad_id,
            'device': device,
            'impressions': imps,
            'clicks': clicks,
            'spend': spend,
        })
        rows_ga.append({
            'date': date,
            'campaign_id': campaign_id,
            'adset_id': adset_id,
            'ad_id': ad_id,
            'device': device,
            'sessions': sessions,
            'transactions': tx,
            'subscriptions': subs,
        })
    return rows_meta, rows_ga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--rows-per-day', type=int, default=120)
    ap.add_argument('--out-meta', required=True)
    ap.add_argument('--out-ga', required=True)
    args = ap.parse_args()

    dates = pd.date_range(args.start, args.end, freq='D').strftime('%Y-%m-%d')
    rng = np.random.default_rng(42)
    meta_rows = []
    ga_rows = []
    for d in dates:
        m, g = gen_day(d, args.rows_per_day, rng)
        meta_rows.extend(m)
        ga_rows.extend(g)

    meta = pd.DataFrame(meta_rows)
    ga = pd.DataFrame(ga_rows)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ga).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(args.out_meta, index=False)
    ga.to_csv(args.out_ga, index=False)
    print(f"Wrote {len(meta)} meta rows and {len(ga)} ga rows")


if __name__ == '__main__':
    main()

