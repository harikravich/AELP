#!/usr/bin/env python3
from __future__ import annotations
"""
Predict probabilities (click>0) using a classifier joblib and write pred_ctr/pred_var.

Usage:
  python pipelines/ctr/predict_proba.py \
    --model artifacts/models/ctr_classifier.joblib \
    --data artifacts/features/marketing_ctr_enhanced.parquet \
    --out artifacts/predictions/ctr_scores_classifier.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    obj = joblib.load(args.model)
    model = obj['model']
    feats = obj['features']

    pth = Path(args.data)
    df = pd.read_parquet(pth) if pth.suffix == '.parquet' else pd.read_csv(pth)
    for c in feats:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feats].fillna(0.0)
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:,1]
    else:
        p = model.predict(X)
    out_df = df[['ad_id']].copy()
    out_df['pred_ctr'] = np.clip(p, 1e-6, 1-1e-6)
    out_df['pred_var'] = 0.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        out_df.to_parquet(out, index=False)
    else:
        out_df.to_csv(out, index=False)
    print(f"Wrote classifier predictions for {len(out_df)} ads to {out}")


if __name__ == '__main__':
    main()

