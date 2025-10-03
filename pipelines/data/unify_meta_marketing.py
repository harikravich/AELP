#!/usr/bin/env python3
from __future__ import annotations
"""
Unify Meta Marketing API outputs (ads.csv + insights.csv) into a CTR training table.

Inputs:
  - --ads artifacts/meta/ads.csv (ad_id, ad_name, creative_id, creative_name)
  - --ins artifacts/meta/insights.csv (date, ad_id, impressions, clicks, spend)

Output columns:
  date, ad_id (creative_id), impressions, clicks, spend, ctr, device, campaign_id, adset_id

Usage:
  python pipelines/data/unify_meta_marketing.py \
    --ads artifacts/meta/ads.csv \
    --ins artifacts/meta/insights.csv \
    --out artifacts/marketing/unified_ctr.parquet
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _safe_div(a, b):
    a = pd.to_numeric(a, errors='coerce')
    b = pd.to_numeric(b, errors='coerce').replace(0, np.nan)
    return (a / b).fillna(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ads', required=True)
    ap.add_argument('--ins', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ads = pd.read_csv(args.ads)
    ins = pd.read_csv(args.ins)
    # Join to get creative_id per ad
    m = ins.merge(ads[['ad_id','creative_id']], on='ad_id', how='left')
    # Use creative_id as the modeling ad_id key
    m['ad_id'] = m['creative_id'].astype(str)
    m['date'] = pd.to_datetime(m['date']).dt.strftime('%Y-%m-%d')
    # Aggregate by date + creative
    g = m.groupby(['date','ad_id'], as_index=False).agg(
        impressions=('impressions','sum'),
        clicks=('clicks','sum'),
        link_clicks=('inline_link_clicks','sum'),
        spend=('spend','sum')
    )
    # CTRs
    g['ctr'] = _safe_div(g['clicks'], g['impressions'])
    g['link_ctr'] = _safe_div(g['link_clicks'], g['impressions'])
    # Fill required columns for downstream
    g['sessions'] = 0.0
    g['transactions'] = 0.0
    g['subscriptions'] = 0.0
    g['spend_per_click'] = _safe_div(g['spend'], g['clicks'])
    g['spend_per_link_click'] = _safe_div(g['spend'], g['link_clicks'])
    g['rpm'] = _safe_div(g['subscriptions'] * 100.0, g['impressions'])
    g['device'] = 'mobile'
    g['campaign_id'] = 'unknown'
    g['adset_id'] = 'unknown'

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == '.parquet':
        g.to_parquet(out, index=False)
    else:
        g.to_csv(out, index=False)
    print(f"Wrote unified CTR table with {len(g)} rows to {out}")


if __name__ == '__main__':
    main()
