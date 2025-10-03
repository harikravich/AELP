#!/usr/bin/env python3
from __future__ import annotations
"""
Forward holdout evaluation for CTR with creative features.
Splits by date (train_end vs holdout_start), trains the CTR regressor
and evaluates Brier/log loss (fractional) and AUC (if both classes present).

Usage:
  python pipelines/validation/ctr_forward_eval.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-26 \
    --holdout-start 2025-09-27 \
    --out artifacts/validation/ctr_forward_marketing.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore


def select_features(df: pd.DataFrame):
    drop = {'date','subscriptions','transactions','ctr','ctr_bin','spend_per_click','clicks','impressions','link_clicks','link_ctr','spend_per_link_click'}
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        if c in {'ad_id','campaign_id','adset_id'}:
            continue
        if df[c].dtype.kind in 'biuf':
            keep.append(c)
    return keep


def build_model(X, y, w):
    if lgb is not None:
        base = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05)
    else:
        base = GradientBoostingRegressor()
    base.fit(X, y, sample_weight=w)
    p_base = np.clip(base.predict(X), 0.0, 1.0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_base, y, sample_weight=w)
    return base, iso


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--holdout-start', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    train = df[df['date'] <= pd.to_datetime(args.train_end)].copy()
    hold = df[df['date'] >= pd.to_datetime(args.holdout_start)].copy()

    ytr = (train.get('link_clicks', train['clicks']) / train['impressions'].replace(0, np.nan)).fillna(0.0).values
    wtr = train['impressions'].astype(float).values
    yte = (hold.get('link_clicks', hold['clicks']) / hold['impressions'].replace(0, np.nan)).fillna(0.0).values
    wte = hold['impressions'].astype(float).values
    feats = select_features(df)
    Xtr = train[feats].fillna(0.0)
    Xte = hold[feats].fillna(0.0)

    base, iso = build_model(Xtr, ytr, wtr)
    p = iso.transform(np.clip(base.predict(Xte), 0.0, 1.0))
    p = np.clip(p, 1e-6, 1-1e-6)
    ll = -np.average(yte * np.log(p) + (1 - yte) * np.log(1 - p), weights=wte)
    brier = float(np.average((p - yte)**2, weights=wte))
    auc = None
    try:
        auc = float(roc_auc_score((yte>0).astype(int), p))
    except Exception:
        auc = None

    out = {
        'rows': int(len(yte)),
        'brier': brier,
        'log_loss': float(ll),
        'auc': auc,
        'train_rows': int(len(ytr)),
        'features': feats
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()
