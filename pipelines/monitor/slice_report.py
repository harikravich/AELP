#!/usr/bin/env python3
from __future__ import annotations
"""
Produce simple monitoring slices for CTR by date and device.

Usage:
  python pipelines/monitor/slice_report.py \
    --data artifacts/features/marketing_ctr_joined.parquet \
    --out artifacts/monitor/slices.json
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    p = Path(args.data)
    df = pd.read_parquet(p) if p.suffix == '.parquet' else pd.read_csv(p)
    df['date'] = pd.to_datetime(df['date'])
    out = {}
    # By date
    by_date = df.groupby('date').agg({'impressions':'sum','clicks':'sum'}).reset_index()
    by_date['ctr'] = (by_date['clicks'] / by_date['impressions'].replace(0, pd.NA)).fillna(0.0)
    out['by_date'] = [
        {'date': d.strftime('%Y-%m-%d'), 'impr': int(i), 'clk': int(c), 'ctr': float(r)}
        for d,i,c,r in zip(by_date['date'], by_date['impressions'], by_date['clicks'], by_date['ctr'])
    ]
    # By device
    dev = 'dev_mobile' if 'dev_mobile' in df.columns else None
    if dev:
        agg = df.groupby(dev).agg({'impressions':'sum','clicks':'sum'}).reset_index()
        agg['ctr'] = (agg['clicks'] / agg['impressions'].replace(0, pd.NA)).fillna(0.0)
        out['by_device'] = [
            {'device': 'mobile' if bool(v) else 'other', 'impr': int(i), 'clk': int(c), 'ctr': float(r)}
            for v,i,c,r in zip(agg[dev], agg['impressions'], agg['clicks'], agg['ctr'])
        ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote slice report to {out_path}")


if __name__ == '__main__':
    main()

