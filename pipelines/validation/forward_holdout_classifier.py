#!/usr/bin/env python3
from __future__ import annotations
"""
Forward holdout evaluation using a classifier trained to predict click>0 with impression weights.

Usage:
  python pipelines/validation/forward_holdout_classifier.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-26 \
    --holdout-start 2025-09-27 \
    --out artifacts/validation/ctr_forward_classifier.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss


def select_features(df: pd.DataFrame):
    drop = {'date','subscriptions','transactions','ctr','ctr_bin','spend_per_click','clicks','impressions','link_clicks','link_ctr','spend_per_link_click','ad_id','campaign_id','adset_id'}
    keep = []
    for c in df.columns:
        if c in drop:
            continue
        if df[c].dtype.kind in 'biuf':
            keep.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--holdout-start', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    train = df[df['date'] <= pd.to_datetime(args.train_end)].copy()
    hold = df[df['date'] >= pd.to_datetime(args.holdout_start)].copy()

    feats = select_features(df)
    Xtr = train[feats].fillna(0.0)
    ytr = (train.get('link_clicks', train['clicks']) > 0).astype(int).values
    wtr = train['impressions'].astype(float).values
    Xte = hold[feats].fillna(0.0)
    yte = (hold.get('link_clicks', hold['clicks']) > 0).astype(int).values
    wte = hold['impressions'].astype(float).values

    if lgb is not None:
        model = lgb.LGBMClassifier(objective='binary', n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    else:
        model = GradientBoostingClassifier()
    model.fit(Xtr, ytr, sample_weight=wtr)

    p = model.predict_proba(Xte)[:,1] if hasattr(model, 'predict_proba') else model.predict(Xte)
    p = np.clip(p, 1e-6, 1-1e-6)
    out = {
        'auc_weighted': float(roc_auc_score(yte, p, sample_weight=wte)),
        'ap_weighted': float(average_precision_score(yte, p, sample_weight=wte)),
        'log_loss_weighted': float(log_loss(yte, p, sample_weight=wte)),
        'rows': int(len(yte)),
        'features': feats
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()
