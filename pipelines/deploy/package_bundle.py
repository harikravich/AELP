#!/usr/bin/env python3
from __future__ import annotations
"""
Package the trained model and TS strategies into a deployable bundle manifest.

Usage:
  python pipelines/deploy/package_bundle.py \
    --model artifacts/models/ctr_creative_marketing.joblib \
    --priors artifacts/priors/priors.json \
    --ts artifacts/priors/ts_strategies_ctr.json \
    --slate artifacts/slates/ad_slate.json \
    --out artifacts/bundle/creative_ctr_bundle.json
"""
import argparse, json
from pathlib import Path
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--priors', required=True)
    ap.add_argument('--ts', required=True)
    ap.add_argument('--slate', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    bundle = {
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'model_path': args.model,
        'priors_path': args.priors,
        'ts_strategies_path': args.ts,
        'slate_path': args.slate,
        'notes': 'Creative-aware CTR bundle with priors and contextual TS strategies.'
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, indent=2))
    print(f"Wrote bundle manifest to {out}")


if __name__ == '__main__':
    main()

