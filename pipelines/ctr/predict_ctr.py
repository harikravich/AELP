#!/usr/bin/env python3
from __future__ import annotations
"""
Predict CTR probabilities for a joined feature table or a list of creatives.

Usage:
  python pipelines/ctr/predict_ctr.py \
    --model artifacts/models/ctr_creative.joblib \
    --data artifacts/features/ctr_joined.parquet \
    --out artifacts/predictions/ctr_scores.parquet
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    obj = joblib.load(args.model)
    models = obj['models']
    feats = obj['features']

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    # Ensure all model features exist in input
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].fillna(0.0)
    ps = []
    for m in models:
        if isinstance(m, dict) and 'base' in m:
            p = m['base'].predict(X)
            # Use isotonic transform to map to [0,1]
            from numpy import clip as _clip
            p = m['iso'].transform(_clip(p, 0.0, 1.0))
        else:
            p = m.predict_proba(X)[:,1] if hasattr(m, 'predict_proba') else m.predict(X)
        ps.append(p)
    p_mean = np.mean(ps, axis=0)
    p_var = np.var(ps, axis=0)
    out_df = df[['ad_id']].copy()
    out_df['pred_ctr'] = p_mean
    out_df['pred_var'] = p_var
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        out_df.to_parquet(out, index=False)
    else:
        out_df.to_csv(out, index=False)
    print(f"Wrote predictions for {len(out_df)} ads to {out}")


if __name__ == '__main__':
    main()
