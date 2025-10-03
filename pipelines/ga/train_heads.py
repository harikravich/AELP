#!/usr/bin/env python3
"""
Train GA heads for c2s (classification) and tps (regression) with calibration.

Usage:
  python pipelines/ga/train_heads.py --data artifacts/unified_training.parquet --outdir artifacts/models/ga_heads
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

try:  # Optional LightGBM
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib  # type: ignore


NUMERIC_FEATURES = [
    'impressions', 'clicks', 'sessions', 'transactions', 'spend',
    'ctr', 'c2s', 'tps', 'spend_per_click', 'rpm'
]


def _prepare(df: pd.DataFrame):
    X = df.copy()
    # target definitions
    y_c2s = (X['c2s'] > 0).astype(int)
    y_tps = X['tps'].astype(float)
    # simple feature subset
    feats = [c for c in NUMERIC_FEATURES if c in X.columns]
    X = X[feats].fillna(0.0)
    return X, y_c2s, y_tps


def train_c2s(X: pd.DataFrame, y: pd.Series):
    if len(np.unique(y)) < 2:
        # Degenerate case; duplicate and force both classes
        X = pd.concat([X, X], ignore_index=True)
        y = pd.concat([y, 1 - y], ignore_index=True)
    if len(X) < 10 or len(y.unique()) < 2:
        Xtr, Xte, ytr, yte = X, X, y, y
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Build base classifier
    if lgb is not None:
        base = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        base.fit(Xtr, ytr)
    else:
        base = GradientBoostingClassifier()
        base.fit(Xtr, ytr)
    # Calibrate
    try:
        calib = CalibratedClassifierCV(base, method='isotonic', cv=3)
        calib.fit(Xtr, ytr)
        model = calib
    except Exception:
        model = base
    try:
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    except Exception:
        auc = float('nan')
    return model, float(auc)


def train_tps(X: pd.DataFrame, y: pd.Series):
    if len(X) < 10:
        Xtr, Xte, ytr, yte = X, X, y, y
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    if lgb is not None:
        reg = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)
        reg.fit(Xtr, ytr)
    else:
        reg = GradientBoostingRegressor()
        reg.fit(Xtr, ytr)
    r2 = r2_score(yte, reg.predict(Xte))
    return reg, float(r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    in_path = Path(args.data)
    if in_path.suffix == '.parquet':
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    X, y_c2s, y_tps = _prepare(df)
    c2s_model, auc = train_c2s(X, y_c2s)
    tps_model, r2 = train_tps(X, y_tps)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': c2s_model, 'features': list(X.columns), 'auc': auc}, outdir / 'c2s.joblib')
    joblib.dump({'model': tps_model, 'features': list(X.columns), 'r2': r2}, outdir / 'tps.joblib')
    print(f"Saved c2s (AUC={auc:.3f}) and tps (R2={r2:.3f}) to {outdir}")


if __name__ == '__main__':
    main()
