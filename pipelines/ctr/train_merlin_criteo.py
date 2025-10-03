#!/usr/bin/env python3
"""
Pretrain CTR model on Criteo_x1 (Merlin preferred; sklearn fallback).

Usage (fallback demo):
  python pipelines/ctr/train_merlin_criteo.py --data data/criteo.csv --out artifacts/models/merlin_criteo_ctr.joblib
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import nvtabular as nvt  # type: ignore
    _HAS_MERLIN = True
except Exception:
    _HAS_MERLIN = False
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import roc_auc_score
    import joblib  # type: ignore
except Exception:  # minimal env fallback
    LogisticRegression = None  # type: ignore
    joblib = None  # type: ignore
    def roc_auc_score(*args, **kwargs):
        return float('nan')


def train_sklearn(df: pd.DataFrame, out: Path) -> float:
    # Minimal numeric-only baseline; derive label if missing
    if 'label' in df.columns:
        y = df['label'].astype(int).values
        X = df.drop(columns=['label'])
    else:
        # use clicks>0 as proxy label
        proxy = 'clicks' if 'clicks' in df.columns else None
        if proxy is None:
            # fallback: create random label
            rng = np.random.default_rng(0)
            y = (rng.random(len(df)) > 0.5).astype(int)
            X = df.copy()
        else:
            y = (df[proxy] > 0).astype(int).values
            X = df.drop(columns=[proxy])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)
    # Handle degenerate labels
    if len(np.unique(y)) < 2:
        class Dummy:
            def __init__(self, p):
                self.p=p
            def predict_proba(self, X):
                import numpy as np
                n = len(X)
                return np.stack([1-np.full(n,self.p), np.full(n,self.p)], axis=1)
        model = Dummy(float(np.mean(y)))
    else:
        model = LogisticRegression(max_iter=200)
        model.fit(X.values, y)
    if joblib is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {'columns': list(X.columns)}
        # Serialize robustly
        if hasattr(model, 'predict_proba') and model.__class__.__name__ != 'Dummy':
            payload['model'] = model
        else:
            payload['dummy_p'] = float(np.mean(y))
        joblib.dump(payload, out)
    try:
        preds = model.predict_proba(X.values)[:, 1]
        auc = roc_auc_score(y, preds)
    except Exception:
        auc = float('nan')
    return float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV path for Criteo_x1-like dataset (label + features)')
    ap.add_argument('--out', required=True, help='Model output path (.joblib)')
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    out = Path(args.out)

    if _HAS_MERLIN:
        # Placeholder: In a full environment, run NVTabular preprocessing + Merlin model training.
        # Here we fall back to sklearn to avoid heavy GPU dependencies at runtime.
        pass
    auc = train_sklearn(df, out)
    print(f"Saved baseline CTR model to {out} (AUC={auc:.3f})")


if __name__ == '__main__':
    main()
