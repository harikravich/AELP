#!/usr/bin/env python3
"""
Forward holdout validation with confidence intervals.

Usage:
  python pipelines/validation/forward_holdout.py --data artifacts/unified_training.parquet --metric auc --out artifacts/validation.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
try:
    from .metrics import brier_score, ece  # when imported as a module
except Exception:
    from pipelines.validation.metrics import brier_score, ece  # when run as script


def bootstrap_ci(metric_values, alpha=0.05):
    lo = np.quantile(metric_values, alpha/2)
    hi = np.quantile(metric_values, 1 - alpha/2)
    return float(lo), float(hi)


def run(df: pd.DataFrame, metric: str = 'auc', n_boot: int = 200, train_end: str | None = None, holdout_start: str | None = None, target: str = 'c2s'):
    # Temporal split by date
    df['date'] = pd.to_datetime(df['date'])
    if train_end and holdout_start:
        train = df[df['date'] <= pd.to_datetime(train_end)]
        hold = df[df['date'] >= pd.to_datetime(holdout_start)]
    else:
        cutoff = df['date'].quantile(0.7)
        train = df[df['date'] <= cutoff]
        hold = df[df['date'] > cutoff]
    # Model on simple features
    feats = ['impressions', 'clicks', 'sessions', 'ctr', 'tps', 'spend_per_click', 'rpm']
    feats = [f for f in feats if f in df.columns]
    Xtr = train[feats].fillna(0.0)
    if target == 'ctr_click':
        ytr = (train['clicks'] > 0).astype(int)
    else:
        ytr = (train['c2s'] > 0).astype(int)
    Xte = hold[feats].fillna(0.0)
    if target == 'ctr_click':
        yte = (hold['clicks'] > 0).astype(int)
    else:
        yte = (hold['c2s'] > 0).astype(int)
    results = {'n_holdout': int(len(yte))}
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        results.update({'auc': None, 'auc_ci': [None, None], 'brier': None, 'brier_ci': [None, None], 'log_loss': None, 'ece': None})
        return results
    clf = LogisticRegression(max_iter=200)
    clf.fit(Xtr, ytr)
    preds = clf.predict_proba(Xte)[:, 1]
    # Metrics
    try:
        results['auc'] = float(roc_auc_score(yte, preds))
    except Exception:
        results['auc'] = None
    results['brier'] = brier_score(yte, preds)
    try:
        results['log_loss'] = float(log_loss(yte, preds, labels=[0,1]))
    except Exception:
        results['log_loss'] = None
    results['ece'] = ece(yte, preds, n_bins=10)

    # Bootstrap CIs
    rng = np.random.default_rng(0)
    aucs, briers = [], []
    n = len(yte)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = yte.iloc[idx], preds[idx]
        try:
            aucs.append(roc_auc_score(yb, pb))
        except Exception:
            pass
        briers.append(brier_score(yb, pb))
    results['auc_ci'] = list(bootstrap_ci(aucs)) if aucs else [None, None]
    results['brier_ci'] = list(bootstrap_ci(briers))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-end', dest='train_end')
    ap.add_argument('--holdout-start', dest='holdout_start')
    ap.add_argument('--target', default='c2s', help='c2s or ctr_click')
    args = ap.parse_args()

    in_path = Path(args.data)
    if in_path.suffix == '.parquet':
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    result = run(df, train_end=args.train_end, holdout_start=args.holdout_start, target=args.target)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Holdout: n={result['n_holdout']} auc={result['auc']} auc_ci={result['auc_ci']} brier={result['brier']:.4f} brier_ci={result['brier_ci']} ece={result['ece']:.4f}")


if __name__ == '__main__':
    main()
