#!/usr/bin/env python3
"""
Compose Beta priors for CTR, c2s, TPS from unified dataset.

Usage:
  python pipelines/priors/compose_beta_priors.py --data artifacts/unified_training.parquet --out artifacts/priors.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd


def beta_from_counts(successes: float, trials: float, alpha0: float = 1.0, beta0: float = 1.0):
    successes = max(0.0, successes)
    failures = max(0.0, trials - successes)
    return float(alpha0 + successes), float(beta0 + failures)


def compose(df: pd.DataFrame, groupby: str | None = None):
    groups = [groupby] if groupby else []
    priors = {}
    grouped = df.groupby(groups) if groups else [("global", df)]
    for key, g in grouped:
        if not isinstance(key, str):
            # pandas returns tuple when grouping by a list, even length-1
            if isinstance(key, tuple) and len(key) == 1:
                key = key[0]
            key = str(key)
        ctr_a, ctr_b = beta_from_counts(g['clicks'].sum(), g['impressions'].sum())
        c2s_a, c2s_b = beta_from_counts(g['subscriptions'].sum(), g['clicks'].sum())
        tps_a, tps_b = beta_from_counts(g['transactions'].sum(), g['sessions'].sum())
        priors[key] = {
            'ctr': {'alpha': ctr_a, 'beta': ctr_b},
            'c2s': {'alpha': c2s_a, 'beta': c2s_b},
            'tps': {'alpha': tps_a, 'beta': tps_b},
        }
    return priors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--groupby', default='campaign_id')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    in_path = Path(args.data)
    if in_path.suffix == '.parquet':
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    priors = compose(df, args.groupby)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(priors, indent=2))
    print(f"Wrote Beta priors to {out}")


if __name__ == '__main__':
    main()
