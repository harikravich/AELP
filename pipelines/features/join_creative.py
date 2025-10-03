#!/usr/bin/env python3
from __future__ import annotations
"""
Join creative features (per ad_id) to the unified dataset and produce a
feature matrix for CTR modeling.

Usage:
  python pipelines/features/join_creative.py \
    --unified artifacts/synth/unified.parquet \
    --creative artifacts/creative/creative_features.parquet \
    --out artifacts/features/ctr_joined.parquet
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


NUMERIC_AGG = [
    'impressions', 'clicks', 'sessions', 'spend', 'ctr', 'spend_per_click', 'rpm'
]


def encode_categoricals(df: pd.DataFrame, cols=('campaign_id','adset_id','ad_id','device')) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str)
    # simple one-hot for device, leave others as ids
    if 'device' in out.columns:
        dummies = pd.get_dummies(out['device'], prefix='dev', dummy_na=False)
        out = pd.concat([out.drop(columns=['device']), dummies], axis=1)
    return out


def bucketize(df: pd.DataFrame, cols=('spend','rpm','ctr'), q=10) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            try:
                out[f'{c}_bin'] = pd.qcut(out[c].fillna(0.0), q=q, labels=False, duplicates='drop')
            except Exception:
                out[f'{c}_bin'] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unified', required=True)
    ap.add_argument('--creative', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    uni_path = Path(args.unified)
    cre_path = Path(args.creative)
    df = pd.read_parquet(uni_path) if uni_path.suffix == '.parquet' else pd.read_csv(uni_path)
    feats = pd.read_parquet(cre_path) if cre_path.suffix == '.parquet' else pd.read_csv(cre_path)

    # Join on ad_id
    if 'ad_id' not in df.columns:
        raise ValueError('unified file missing ad_id')
    df['ad_id'] = df['ad_id'].astype(str)
    if 'ad_id' in feats.columns and len(feats) > 0:
        feats['ad_id'] = feats['ad_id'].astype(str)
        j = df.merge(feats, how='left', on='ad_id')
    else:
        j = df.copy()

    # Clean numeric
    for c in NUMERIC_AGG:
        if c in j.columns:
            j[c] = pd.to_numeric(j[c], errors='coerce').fillna(0.0)

    j = encode_categoricals(j)
    j = bucketize(j)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        j.to_parquet(out, index=False)
    else:
        j.to_csv(out, index=False)
    print(f"Wrote joined features with {len(j)} rows to {out}")


if __name__ == '__main__':
    main()
