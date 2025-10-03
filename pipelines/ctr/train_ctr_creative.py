#!/usr/bin/env python3
from __future__ import annotations
"""
Train a creative-aware CTR model with calibration and uncertainty.

Targets: p(click) with impression-weighted loss. Uses LightGBM if available,
falls back to GradientBoosting + isotonic calibration. Uncertainty estimated
via bootstrap ensembles.

Usage:
  python pipelines/ctr/train_ctr_creative.py \
    --data artifacts/features/ctr_joined.parquet \
    --out artifacts/models/ctr_creative.joblib
"""
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def select_features(df: pd.DataFrame) -> List[str]:
    drop_cols = {
        'date', 'subscriptions', 'transactions',
        # target/leakage columns
        'ctr', 'ctr_bin', 'clicks', 'impressions', 'spend_per_click',
        'link_clicks', 'link_ctr', 'spend_per_link_click'
    }
    cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if c in {'ad_id','campaign_id','adset_id'}:
            continue
        if df[c].dtype.kind in 'biuf':
            cols.append(c)
    return cols


def build_model(X: pd.DataFrame, y: np.ndarray, w: np.ndarray):
    # Regression on CTR rate with isotonic calibration to [0,1]
    if lgb is not None:
        base = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    else:
        base = GradientBoostingRegressor()
    base.fit(X, y, sample_weight=w)
    # Fit isotonic calibration mapping base predictions -> probability
    p_base = np.clip(base.predict(X), 0.0, 1.0)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(p_base, y, sample_weight=w)
    return {'base': base, 'iso': iso}


def evaluate(model, Xv, yv, wv) -> Dict[str, Any]:
    if isinstance(model, dict) and 'base' in model:
        p = model['base'].predict(Xv)
        p = model['iso'].transform(np.clip(p, 0.0, 1.0))
    else:
        p = model.predict(Xv)
    p = np.clip(p, 1e-6, 1-1e-6)
    # Weighted Bernoulli negative log-likelihood with fractional labels
    ll = -np.average(yv * np.log(p) + (1 - yv) * np.log(1 - p), weights=wv)
    metrics = {
        'rows': int(len(yv)),
        'log_loss': float(ll),
        'brier': float(np.average((p - yv)**2, weights=wv)),
    }
    try:
        metrics['auc'] = float(roc_auc_score(yv, p))
    except Exception:
        metrics['auc'] = None
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_ensem', type=int, default=5)
    args = ap.parse_args()

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    # Target: CTR rate; weights=impressions
    # Use link CTR
    if 'link_clicks' in df.columns:
        y = (df['link_clicks'] / df['impressions'].replace(0, np.nan)).fillna(0.0).values
    else:
        y = (df['clicks'] / df['impressions'].replace(0, np.nan)).fillna(0.0).values
    w = df.get('impressions', pd.Series(np.ones(len(df)))).astype(float).values
    feats = select_features(df)
    X = df[feats].fillna(0.0)
    Xtr, Xv, ytr, yv, wtr, wv = train_test_split(X, y, w, test_size=0.2, random_state=42)

    models = []
    rng = np.random.default_rng(0)
    for i in range(args.n_ensem):
        idx = rng.integers(0, len(Xtr), size=len(Xtr))
        m = build_model(Xtr.iloc[idx], ytr[idx], wtr[idx])
        models.append(m)
    # Average probabilities for validation metrics
    ps = []
    for m in models:
        if isinstance(m, dict) and 'base' in m:
            p = m['base'].predict(Xv)
            p = m['iso'].transform(np.clip(p, 0.0, 1.0))
        else:
            p = m.predict(Xv)
        ps.append(p)
    p_mean = np.mean(ps, axis=0)
    p_var = np.var(ps, axis=0)
    p_mean = np.clip(p_mean, 1e-6, 1-1e-6)
    ll = -np.average(yv * np.log(p_mean) + (1 - yv) * np.log(1 - p_mean), weights=wv)
    val_metrics = {
        'rows': int(len(yv)),
        'log_loss': float(ll),
        'brier': float(np.average((p_mean - yv)**2, weights=wv)),
        'auc': float(roc_auc_score((yv>0).astype(int), p_mean)) if len(np.unique((yv>0).astype(int))) > 1 else None,
        'mean_var': float(p_var.mean())
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': models, 'features': feats, 'val_metrics': val_metrics}, out)
    print({'val': val_metrics, 'out': str(out)})


if __name__ == '__main__':
    main()
