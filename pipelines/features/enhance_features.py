#!/usr/bin/env python3
from __future__ import annotations
"""
Enhance joined CTR dataset with leakage-safe priors (ad_id and campaign_id) and temporal features.

Usage:
  python pipelines/features/enhance_features.py \
    --in artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-26 \
    --out artifacts/features/marketing_ctr_enhanced.parquet
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def beta_from_counts(successes: float, trials: float, a0: float = 1.0, b0: float = 1.0):
    successes = max(0.0, successes)
    failures = max(0.0, trials - successes)
    return a0 + successes, b0 + failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.inp)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    train = df[df['date'] <= pd.to_datetime(args.train_end)].copy()

    # Priors per ad_id using train-only counts
    ag = train.groupby('ad_id').agg(impr=('impressions','sum'), clk=('clicks','sum')).reset_index()
    ag[['alpha_ad','beta_ad']] = ag.apply(lambda r: pd.Series(beta_from_counts(r['clk'], r['impr'])), axis=1)
    ag['prior_ctr_ad'] = ag['alpha_ad'] / (ag['alpha_ad'] + ag['beta_ad'])
    df = df.merge(ag[['ad_id','alpha_ad','beta_ad','prior_ctr_ad']], on='ad_id', how='left')

    # Priors per campaign_id if present
    if 'campaign_id' in df.columns:
        cg = train.groupby('campaign_id').agg(impr=('impressions','sum'), clk=('clicks','sum')).reset_index()
        cg[['alpha_camp','beta_camp']] = cg.apply(lambda r: pd.Series(beta_from_counts(r['clk'], r['impr'])), axis=1)
        cg['prior_ctr_camp'] = cg['alpha_camp'] / (cg['alpha_camp'] + cg['beta_camp'])
        df = df.merge(cg[['campaign_id','alpha_camp','beta_camp','prior_ctr_camp']], on='campaign_id', how='left')

    # Temporal features
    df['dow'] = df['date'].dt.weekday.astype(int)
    first_seen = df.groupby('ad_id')['date'].transform('min')
    df['age_days'] = (df['date'] - first_seen).dt.days.clip(lower=0).astype(int)

    # Fill NaNs in priors with global prior from train counts
    g_impr = float(train['impressions'].sum()); g_clk = float(train['clicks'].sum())
    a_g, b_g = beta_from_counts(g_clk, g_impr)
    prior_g = a_g / (a_g + b_g) if (a_g + b_g) > 0 else 0.0
    df['prior_ctr_ad'] = df['prior_ctr_ad'].fillna(prior_g)
    if 'prior_ctr_camp' in df.columns:
        df['prior_ctr_camp'] = df['prior_ctr_camp'].fillna(prior_g)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote enhanced features to {out} with shape {df.shape}")


if __name__ == '__main__':
    main()

