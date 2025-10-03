#!/usr/bin/env python3
"""
Slate Builder: rank assets with diversity/novelty and step-goal weighting.

Inputs:
- creative_features parquet/csv (optional)
- unified dataset for baseline metrics (optional)

Usage:
  python pipelines/slate/build_slate.py --assets artifacts/landers --features artifacts/creative_features.parquet --k 5 --out artifacts/slates/lander_slate.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def cosine(a, b):
    if a is None or b is None:
        return 0.0
    a = np.asarray(a); b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_slate(catalog: pd.DataFrame, k: int = 5,
                w_rel: float = 0.6, w_div: float = 0.3, w_nov: float = 0.1) -> List[Dict[str, Any]]:
    # Precompute vectors
    clip_cols = [c for c in catalog.columns if c.startswith('clip_')]
    selected = []
    remaining = catalog.copy()
    novelty = (1.0 / (1.0 + remaining.groupby('slug')['slug'].transform('count'))).values
    remaining['novelty'] = novelty
    while len(selected) < min(k, len(remaining)):
        scores = []
        view = remaining.reset_index(drop=True)
        vecs = view[clip_cols].values if clip_cols else None
        for i, row in view.iterrows():
            rel = float(row.get('pred_score', 0.5))
            nov = float(row.get('novelty', 0.5))
            if not selected or vecs is None:
                div = 1.0
            else:
                # diversity: 1 - max cosine similarity with selected
                vec = vecs[i] if vecs is not None else None
                max_sim = 0.0
                for s in selected:
                    j = s['__idx']
                    max_sim = max(max_sim, cosine(vec, vecs[j]) if vecs is not None else 0.0)
                div = 1.0 - max_sim
            score = w_rel * rel + w_div * div + w_nov * nov
            scores.append((score, i))
        _, best_i = max(scores, key=lambda x: x[0])
        rec = view.iloc[best_i].to_dict()
        rec['__idx'] = best_i
        selected.append(rec)
        remaining = view.drop(view.index[best_i])
    for r in selected:
        r.pop('__idx', None)
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--assets', required=True, help='Directory with .html landers or images to slate')
    ap.add_argument('--features', required=False, help='Parquet/CSV creative features (optional)')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    asset_dir = Path(args.assets)
    records = []
    exts = ['*.html', '*.png', '*.jpg', '*.jpeg']
    for ext in exts:
        for p in asset_dir.rglob(ext):
            records.append({'slug': p.stem, 'path': str(p), 'pred_score': 0.5})
    catalog = pd.DataFrame.from_records(records)
    if args.features and Path(args.features).exists():
        feats = pd.read_parquet(args.features) if args.features.endswith('.parquet') else pd.read_csv(args.features)
        if 'asset_path' in feats.columns:
            catalog = catalog.merge(feats, how='left', left_on='path', right_on='asset_path')
    slate = build_slate(catalog, k=args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'items': slate}, indent=2))
    print(f"Wrote slate of {len(slate)} items to {out}")


if __name__ == '__main__':
    main()
