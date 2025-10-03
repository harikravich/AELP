#!/usr/bin/env python3
from __future__ import annotations
"""
Contextual Thompson Sampling with linear posterior (LinTS) and novelty prior.

Inputs:
  - features table with per-ad features and a baseline prior (alpha,beta) optional
  - predictions with mean and variance (pred_ctr, pred_var)

Outputs:
  - JSON strategies for each ad_id with combined prior and contextual mean/uncertainty

Usage:
  python pipelines/bandit/contextual_ts.py \
    --features artifacts/features/ctr_joined.parquet \
    --preds artifacts/predictions/ctr_scores.parquet \
    --priors artifacts/priors/priors.json \
    --out artifacts/priors/ts_strategies_ctr.json
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def novelty_boost(emb: np.ndarray, catalog: np.ndarray, k: int = 20) -> float:
    if catalog.size == 0 or emb.size == 0:
        return 0.0
    # cosine distance to nearest neighbor set
    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    sims = [cosine(emb, e) for e in catalog]
    sims = sorted(sims, reverse=True)[:k]
    mean_sim = np.mean(sims) if sims else 0.0
    return float(max(0.0, 1.0 - mean_sim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--preds', required=True)
    ap.add_argument('--priors', required=False)
    ap.add_argument('--out', required=True)
    ap.add_argument('--novelty-weight', type=float, default=0.05)
    args = ap.parse_args()

    feats_path = Path(args.features)
    preds_path = Path(args.preds)
    feats = pd.read_parquet(feats_path) if feats_path.suffix == '.parquet' else pd.read_csv(feats_path)
    preds = pd.read_parquet(preds_path) if preds_path.suffix == '.parquet' else pd.read_csv(preds_path)
    df = feats.merge(preds, on='ad_id', how='inner')

    # Embedding matrix for novelty: prefer CLIP; fallback to numeric creative features
    clip_cols = [c for c in df.columns if c.startswith('clip_')]
    if clip_cols:
        emb_mat = df[clip_cols].fillna(0.0).values
    else:
        alt_cols = [c for c in df.columns if c.startswith('obj_') and df[c].dtype.kind in 'biuf']
        alt_cols += [c for c in ['sharpness','width','height','aspect_ratio'] if c in df.columns]
        if alt_cols:
            mat = df[alt_cols].fillna(0.0).values
            # normalize rows to unit length to approximate cosine
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            emb_mat = mat / norms
        else:
            emb_mat = np.zeros((len(df), 0))

    pri = {}
    if args.priors and Path(args.priors).exists():
        pri = json.loads(Path(args.priors).read_text())

    # Build strategies per ad
    strategies = {}
    for i, row in df.iterrows():
        ad = str(row['ad_id'])
        mu = float(row['pred_ctr'])
        var = float(row.get('pred_var', 0.01))
        # Map prior: use ad_id specific if present, else campaign_id, else global
        prior = {'alpha': 1.0, 'beta': 1.0}
        if ad in pri:
            prior = pri[ad].get('ctr', prior)
        elif 'campaign_id' in row and str(row['campaign_id']) in pri:
            prior = pri[str(row['campaign_id'])].get('ctr', prior)
        elif 'global' in pri:
            prior = pri['global'].get('ctr', prior)

        # Novelty
        nov = 0.0
        if emb_mat.shape[1] > 0:
            nov = novelty_boost(emb_mat[i], np.delete(emb_mat, i, axis=0))

        # Combine: prior mean with contextual mean and novelty
        prior_mean = prior['alpha'] / (prior['alpha'] + prior['beta'])
        combined = 0.7 * mu + 0.25 * prior_mean + args.novelty_weight * nov

        strategies[ad] = {
            'alpha': float(prior['alpha']),
            'beta': float(prior['beta']),
            'pred_mu': mu,
            'pred_var': var,
            'novelty': nov,
            'combined_score': float(combined)
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strategies, indent=2))
    print(f"Wrote contextual TS strategies for {len(strategies)} ads to {out}")


if __name__ == '__main__':
    main()
