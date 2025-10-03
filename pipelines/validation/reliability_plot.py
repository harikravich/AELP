#!/usr/bin/env python3
from __future__ import annotations
"""
Generate reliability (calibration) plots for CTR predictions.

Loads a trained joblib model bundle and a joined features dataset,
computes predictions on a holdout date range, and saves:
  - artifacts/validation/reliability.png
  - artifacts/validation/calibration_table.csv

Usage:
  python pipelines/validation/reliability_plot.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --model artifacts/models/ctr_creative_marketing.joblib \
    --holdout-start 2025-09-27 \
    --outdir artifacts/validation
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None


def predict_bundle(bundle, X: pd.DataFrame) -> np.ndarray:
    preds = []
    for m in bundle['models']:
        base = m['base'] if isinstance(m, dict) else m
        p = base.predict(X)
        if isinstance(m, dict) and 'iso' in m:
            p = m['iso'].transform(np.clip(p, 0.0, 1.0))
        preds.append(p)
    P = np.mean(preds, axis=0)
    return np.clip(P, 1e-6, 1-1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--holdout-start', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--bins', type=int, default=20)
    args = ap.parse_args()

    dpath = Path(args.data)
    df = pd.read_parquet(dpath) if dpath.suffix == '.parquet' else pd.read_csv(dpath)
    df['date'] = pd.to_datetime(df['date'])
    hold = df[df['date'] >= pd.to_datetime(args.holdout_start)].copy()

    bundle = joblib.load(args.model)
    feat_cols = bundle['features']
    X = hold[feat_cols].fillna(0.0)
    y = (hold['clicks'] / hold['impressions'].replace(0, np.nan)).fillna(0.0).values
    w = hold['impressions'].astype(float).values
    p = predict_bundle(bundle, X)

    # Weighted calibration table in quantile bins of predictions
    q = np.linspace(0, 1, args.bins + 1)
    edges = np.quantile(p, q)
    # Deduplicate edges if constant preds
    edges = np.unique(edges)
    idx = np.digitize(p, edges, right=True)
    rows = []
    for k in range(1, len(edges) + 1):
        mask = idx == k
        if not np.any(mask):
            continue
        pw = w[mask]
        rows.append({
            'bin': k,
            'p_mean': float(np.average(p[mask], weights=pw)),
            'y_mean': float(np.average(y[mask], weights=pw)),
            'count': int(mask.sum()),
            'impr': float(pw.sum())
        })
    cal = pd.DataFrame(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cal_path = outdir / 'calibration_table.csv'
    cal.to_csv(cal_path, index=False)

    if plt is not None:
        # Plot
        plt.figure(figsize=(5,5))
        plt.plot([0,1], [0,1], 'k--', label='Ideal')
        plt.plot(cal['p_mean'], cal['y_mean'], 'o-', label='Model')
        plt.xlabel('Predicted CTR')
        plt.ylabel('Observed CTR')
        plt.title('Reliability Diagram (Impr-weighted)')
        plt.legend()
        png_path = outdir / 'reliability.png'
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        print(f"Wrote {cal_path} and {png_path}")
    else:
        # Write a simple ASCII plot as fallback
        txt_path = outdir / 'reliability.txt'
        with txt_path.open('w') as f:
            f.write('Reliability Diagram (ASCII)\n')
            for _, r in cal.iterrows():
                f.write(f"bin={int(r['bin'])} p_mean={r['p_mean']:.6f} y_mean={r['y_mean']:.6f} impr={int(r['impr'])}\n")
        print(f"Wrote {cal_path} and {txt_path}")


if __name__ == '__main__':
    main()
