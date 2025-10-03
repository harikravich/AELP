#!/usr/bin/env python3
from __future__ import annotations
"""
Train a ranking-oriented CTR model (LambdaRank) to improve AUC/ordering.

Groups by date (or date+campaign if present) and optimizes pairwise ranking via LightGBM Ranker.

Usage:
  python pipelines/ctr/train_ctr_ranker.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --train-end 2025-09-26 \
    --holdout-start 2025-09-27 \
    --out artifacts/models/ctr_ranker.joblib
"""
import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import joblib

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None  # type: ignore


def select_features(df: pd.DataFrame) -> List[str]:
    drop = {
        'date','subscriptions','transactions',
        'ctr','ctr_bin','spend_per_click','clicks','impressions','ad_id','campaign_id','adset_id'
    }
    feats: List[str] = []
    for c in df.columns:
        if c in drop:
            continue
        if df[c].dtype.kind in 'biuf':
            feats.append(c)
    return feats


def build_groups(df: pd.DataFrame) -> np.ndarray:
    # group by date, or date+campaign_id if present
    if 'campaign_id' in df.columns:
        key = df['date'].astype(str) + '|' + df['campaign_id'].astype(str)
    else:
        key = df['date'].astype(str)
    sizes = key.value_counts().reindex(key).fillna(0).astype(int).values
    # convert to group sizes for LightGBM
    groups = []
    i = 0
    while i < len(sizes):
        n = sizes[i]
        groups.append(int(n))
        i += n
    return np.array(groups, dtype=int)


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

    # Labels and weights
    ytr = (train['clicks'] / train['impressions'].replace(0, np.nan)).fillna(0.0).values
    wtr = train['impressions'].astype(float).values
    yte = (hold['clicks'] / hold['impressions'].replace(0, np.nan)).fillna(0.0).values
    wte = hold['impressions'].astype(float).values

    feats = select_features(df)
    Xtr = train[feats].fillna(0.0)
    Xte = hold[feats].fillna(0.0)

    if lgb is None:
        raise SystemExit('lightgbm not available; install it to use the ranker')

    # Build group sizes on the training set
    train_sorted = train.sort_values(['date','campaign_id'] if 'campaign_id' in train.columns else ['date'])
    Xtr = train_sorted[feats].fillna(0.0)
    ytr = (train_sorted['clicks'] / train_sorted['impressions'].replace(0, np.nan)).fillna(0.0).values
    wtr = train_sorted['impressions'].astype(float).values
    group_sizes = []
    if 'campaign_id' in train_sorted.columns:
        grp = train_sorted.groupby(['date','campaign_id']).size()
        group_sizes = grp.astype(int).tolist()
    else:
        grp = train_sorted.groupby(['date']).size()
        group_sizes = grp.astype(int).tolist()

    model = lgb.LGBMRanker(
        objective='lambdarank',
        n_estimators=600,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=20,
    )
    model.fit(Xtr, ytr, group=group_sizes, sample_weight=wtr)

    # Predict on holdout
    p = model.predict(Xte)
    # Ranking metrics
    from sklearn.metrics import roc_auc_score
    try:
        auc = float(roc_auc_score((yte>0).astype(int), p, sample_weight=wte))
    except Exception:
        auc = None

    # Pairwise win-rate within date(+campaign)
    def pairwise_winrate(dfh: pd.DataFrame, scores: np.ndarray) -> float:
        dfh = dfh.copy()
        dfh['score'] = scores
        if 'campaign_id' in dfh.columns:
            keys = ['date','campaign_id']
        else:
            keys = ['date']
        wins = 0
        total = 0
        for _, g in dfh.groupby(keys):
            arr = g[['score']].values.ravel()
            lab = (g['clicks'] / g['impressions'].replace(0, np.nan)).fillna(0.0).values
            n = len(arr)
            for i in range(n):
                for j in range(i+1, n):
                    if lab[i] == lab[j]:
                        continue
                    total += 1
                    if (arr[i] > arr[j]) == (lab[i] > lab[j]):
                        wins += 1
        return float(wins / total) if total else 0.0

    wr = pairwise_winrate(hold, p)
    out = {
        'features': feats,
        'auc_weighted': auc,
        'pairwise_winrate': wr,
        'n_holdout': int(len(yte)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': feats, 'metrics': out}, args.out)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

