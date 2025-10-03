#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluate ranking quality on holdout: impression-weighted ROC AUC, PR-AUC, pairwise win-rate, and NDCG@k.

Usage:
  python pipelines/validation/ranking_eval.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-26 \
    --holdout-start 2025-09-27 \
    --model artifacts/models/ctr_ranker.joblib \
    --out artifacts/validation/ranking_eval.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    idx = np.argsort(-y_score)[:k]
    gains = (2**y_true[idx] - 1)
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    # ideal
    idx_i = np.argsort(-y_true)[:k]
    gains_i = (2**y_true[idx_i] - 1)
    idcg = np.sum(gains_i / discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-end', required=True)
    ap.add_argument('--holdout-start', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    hold = df[df['date'] >= pd.to_datetime(args.holdout_start)].copy()
    hold = hold.reset_index(drop=True)
    hold['__pos'] = np.arange(len(hold))

    bundle = joblib.load(args.model)
    model = bundle['model']
    feats = bundle['features']
    X = hold[feats].fillna(0.0)
    y = (hold.get('link_clicks', hold['clicks']) / hold['impressions'].replace(0, np.nan)).fillna(0.0).values
    w = hold['impressions'].astype(float).values
    s = model.predict(X)

    # Weighted ROC AUC (binary >0 vs 0)
    yb = (hold.get('link_clicks', hold['clicks']).values > 0).astype(int)
    try:
        auc_w = float(roc_auc_score(yb, s, sample_weight=w))
    except Exception:
        auc_w = None
    # PR-AUC (AP)
    try:
        ap_w = float(average_precision_score(yb, s, sample_weight=w))
    except Exception:
        ap_w = None

    # Pairwise win-rate per date
    wins = 0; total = 0
    for _, g in hold.groupby('date'):
        pos = g['__pos'].values
        sc = s[pos]
        yy = y[pos]
        n = len(g)
        for i in range(n):
            for j in range(i+1, n):
                if yy[i] == yy[j]:
                    continue
                total += 1
                if (sc[i] > sc[j]) == (yy[i] > yy[j]):
                    wins += 1
    wr = float(wins/total) if total else 0.0

    # NDCG@5, @10 averaged by date
    nd5 = []
    nd10 = []
    for _, g in hold.groupby('date'):
        pos = g['__pos'].values
        sc = s[pos]
        yy = y[pos]
        if len(g) >= 2:
            nd5.append(ndcg_at_k(yy, sc, 5))
            nd10.append(ndcg_at_k(yy, sc, 10))
    out = {
        'auc_weighted': auc_w,
        'ap_weighted': ap_w,
        'pairwise_winrate': wr,
        'ndcg@5_mean': float(np.mean(nd5)) if nd5 else None,
        'ndcg@10_mean': float(np.mean(nd10)) if nd10 else None,
        'rows': int(len(hold))
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == '__main__':
    main()
