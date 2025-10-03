#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import roc_auc_score, log_loss


def make_tokens(df: pd.DataFrame) -> list[list[str]]:
    bins = {
        'impressions': np.array([0, 100, 500, 1000, 5000, 10000, 50000, 1e9]),
        'clicks': np.array([0, 1, 2, 5, 10, 20, 50, 1e9]),
        'sessions': np.array([0, 10, 20, 50, 100, 200, 500, 1e9]),
        'ctr': np.linspace(0.0, 0.5, 11),
        'spend_per_click': np.array([0, 0.2, 0.5, 1, 2, 5, 10, 1e9]),
        'rpm': np.array([0, 1, 5, 10, 20, 50, 100, 1e9]),
    }
    recs = []
    for row in df.to_dict(orient='records'):
        toks = []
        for key in ('campaign_id','adset_id','ad_id','device'):
            if key in row and pd.notna(row[key]):
                toks.append(f"{key}={row[key]}")
        # bucketize selected numeric features
        for k, b in bins.items():
            if k in row and pd.notna(row[k]):
                idx = int(np.digitize([float(row[k])], b)[0])
                toks.append(f"{k}_bin={idx}")
        recs.append(toks)
    return recs


def weighted_brier(y, p, w):
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.asarray(w, float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * (p - y)**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--unified', required=True)
    ap.add_argument('--holdout-start', required=True, help='YYYY-MM-DD inclusive')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    obj = joblib.load(args.model)
    clf = obj['clf']
    n_features = obj['hash_bits']
    hasher = FeatureHasher(n_features=n_features, input_type='string')

    path = Path(args.unified)
    df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    hold = df[df['date'] >= pd.to_datetime(args.holdout_start)].copy()
    # label: click>0 approximates Avazu click
    y = (hold['clicks'] > 0).astype(int).values
    w = hold.get('impressions', pd.Series(np.ones(len(hold)))).astype(float).values
    toks = make_tokens(hold)
    X = hasher.transform(toks)
    p = clf.predict_proba(X)[:,1]

    out = {
        'rows': int(len(y)),
        'weighted_brier': weighted_brier(y, p, w),
        'log_loss': float(log_loss(y, p, labels=[0,1], sample_weight=w)),
    }
    # AUC (unweighted)
    try:
        out['auc'] = float(roc_auc_score(y, p))
    except Exception:
        out['auc'] = None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(pd.Series(out).to_json(indent=2))
    print(out)


if __name__ == '__main__':
    main()

