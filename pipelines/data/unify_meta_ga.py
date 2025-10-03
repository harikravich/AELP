#!/usr/bin/env python3
"""
Unify Meta Ads + GA4 datasets into a single training table.

Outputs columns (example):
- date, campaign_id, adset_id, ad_id, creative_id, device, hour
- impressions, clicks, sessions, transactions
- ctr, c2s, tps
- context_* (derived)

CLI:
  python pipelines/data/unify_meta_ga.py \
    --meta data/meta_daily.csv \
    --ga data/ga_daily.csv \
    --out artifacts/unified_training.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _safe_div(a, b):
    return (a / b).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def unify(meta_df: pd.DataFrame, ga_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize keys
    meta = meta_df.copy()
    ga = ga_df.copy()

    for df in (meta, ga):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date.astype(str)

    # Minimal keys for join
    join_keys = ['date', 'campaign_id', 'adset_id', 'ad_id', 'device']
    for k in join_keys:
        if k not in meta.columns:
            meta[k] = 'unknown'
        if k not in ga.columns:
            ga[k] = 'unknown'

    meta_agg = meta.groupby(join_keys, as_index=False).agg(
        impressions=('impressions', 'sum'),
        clicks=('clicks', 'sum'),
        spend=('spend', 'sum'),
    )

    ga_agg = ga.groupby(join_keys, as_index=False).agg(
        sessions=('sessions', 'sum'),
        transactions=('transactions', 'sum'),
        subscriptions=('subscriptions', 'sum'),
    )

    df = pd.merge(meta_agg, ga_agg, on=join_keys, how='outer').fillna(0)

    # KPIs
    df['ctr'] = _safe_div(df['clicks'], df['impressions'])
    df['c2s'] = _safe_div(df['subscriptions'], df['clicks'])
    df['tps'] = _safe_div(df['transactions'], df['sessions'])

    # Context features
    df['spend_per_click'] = _safe_div(df['spend'], df['clicks'].replace(0, np.nan))
    df['rpm'] = _safe_div(df['subscriptions'] * 100.0, df['impressions'])  # subs per 100 impressions

    # Sorting for reproducibility
    df = df.sort_values(join_keys).reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta', type=str, required=True)
    ap.add_argument('--ga', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta)
    ga = pd.read_csv(args.ga)
    out_df = unify(meta, ga)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == '.parquet':
        out_df.to_parquet(out_path, index=False)
    else:
        out_df.to_csv(out_path, index=False)
    print(f"Wrote unified dataset to {out_path} with {len(out_df)} rows")


if __name__ == '__main__':
    main()

