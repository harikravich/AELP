#!/usr/bin/env python3
from __future__ import annotations
"""
Create a catalog feature table for novel creatives (no historical aggregates).
It fills required aggregate columns with zeros and keeps creative features.

Usage:
  python pipelines/features/catalog_from_creatives.py \
    --creative artifacts/creative/creative_features.parquet \
    --out artifacts/features/novel_catalog.parquet
"""
import argparse
from pathlib import Path
import pandas as pd

try:
    from .join_creative import encode_categoricals, bucketize
except Exception:
    from pipelines.features.join_creative import encode_categoricals, bucketize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--creative', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.creative)
    feats = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df = feats.copy()
    # Ensure mandatory columns exist
    for c in ['impressions','clicks','sessions','spend','ctr','spend_per_click','rpm','campaign_id','adset_id','device']:
        if c not in df.columns:
            if c in ('campaign_id','adset_id'):
                df[c] = 'novel'
            elif c == 'device':
                df[c] = 'mobile'
            else:
                df[c] = 0.0
    df = encode_categoricals(df)
    df = bucketize(df)
    # ad_id present in feats
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote novel catalog with {len(df)} rows to {out}")


if __name__ == '__main__':
    main()
