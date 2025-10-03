#!/usr/bin/env python3
from __future__ import annotations
"""
Train a classifier to optimize AUC on binary clicks (>0) with impression weights.

Usage:
  python pipelines/ctr/train_ctr_classifier.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --out artifacts/models/ctr_classifier.joblib
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import train_test_split


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
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    y = (df.get('link_clicks', df['clicks']) > 0).astype(int).values
    w = df['impressions'].astype(float).values
    feats = select_features(df)
    X = df[feats].fillna(0.0)
    Xtr, Xv, ytr, yv, wtr, wv = train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)

    if lgb is not None:
        model = lgb.LGBMClassifier(objective='binary', n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    else:
        model = GradientBoostingClassifier()
    model.fit(Xtr, ytr, sample_weight=wtr)
    pv = model.predict_proba(Xv)[:,1] if hasattr(model, 'predict_proba') else model.predict(Xv)
    metrics = {
        'auc': float(roc_auc_score(yv, pv, sample_weight=wv)),
        'ap': float(average_precision_score(yv, pv, sample_weight=wv)),
        'log_loss': float(log_loss(yv, np.clip(pv,1e-6,1-1e-6), sample_weight=wv)),
        'rows': int(len(yv)),
        'features': feats
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': feats, 'val': metrics}, out)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
