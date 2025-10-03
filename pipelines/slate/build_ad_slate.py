#!/usr/bin/env python3
from __future__ import annotations
"""
Build a ranked slate of ad creatives from Contextual TS strategies and manifest.

Usage:
  python pipelines/slate/build_ad_slate.py \
    --ts artifacts/priors/ts_strategies_ctr.json \
    --manifest assets/meta_creatives/ads_manifest.csv \
    --k 30 \
    --out artifacts/slates/ad_slate.json
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ts = json.loads(Path(args.ts).read_text())
    man = pd.read_csv(args.manifest)
    man['ad_id'] = man['ad_id'].astype(str)
    rows = []
    for ad, s in ts.items():
        rows.append({
            'ad_id': ad,
            'score': float(s.get('combined_score', s.get('pred_mu', 0.0))),
            'novelty': float(s.get('novelty', 0.0)),
            'alpha': float(s.get('alpha', 1.0)),
            'beta': float(s.get('beta', 1.0)),
        })
    df = pd.DataFrame(rows)
    slate = df.sort_values('score', ascending=False).head(args.k)
    slate = slate.merge(man[['ad_id','asset_path','headline','primary_text','destination_url']], on='ad_id', how='left')
    slate = slate.fillna('')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'items': slate.to_dict(orient='records')}, indent=2))
    print(f"Wrote slate of {len(slate)} ads to {out}")


if __name__ == '__main__':
    main()
