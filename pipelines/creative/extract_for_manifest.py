#!/usr/bin/env python3
from __future__ import annotations
"""
Extract creative features for assets listed in a CSV manifest with columns:
  ad_id,asset_path

Writes a Parquet/CSV with ad_id and extracted features.

Usage:
  python pipelines/creative/extract_for_manifest.py \
    --manifest assets/demo_creatives/ads_manifest.csv \
    --out artifacts/creative/creative_features.parquet
"""
import argparse
from pathlib import Path
import pandas as pd

try:
    # when run as module
    from .extract_features import extract_from_path
except Exception:
    # when run as script
    from pipelines.creative.extract_features import extract_from_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    records = []
    for _, row in man.iterrows():
        apath = row.get('asset_path')
        if not isinstance(apath, str) or not apath:
            continue
        p = Path(apath)
        if not p.exists():
            continue
        feats = extract_from_path(p)
        feats['ad_id'] = str(row['ad_id'])
        records.append(feats)
    df = pd.DataFrame.from_records(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"Wrote creative features for {len(df)} assets to {out}")


if __name__ == '__main__':
    main()
