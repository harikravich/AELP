#!/usr/bin/env python3
from __future__ import annotations
"""
Backtest an ensemble (Auction + Thompson) on campaign creative sets with labels.

Inputs:
- AELP2/reports/creative/*.json  (per-campaign: creative_id, sim_score, actual_label)
- BigQuery meta_ad_performance (for CTR/CVR evidence) via backtest_auction_accuracy helpers

Method:
- For each campaign with items:
  * Build labels: actual_label
  * TS ranking: sort by descending sim_score
  * Auction CAC: predict via AuctionGym + CPC model (if present), same as backtest_auction_accuracy
  * Compute ranks: `rank_ts` (1..N), `rank_au` (1..N)
  * For weights w in {0.0, 0.05, ..., 1.0}: combined_rank = w*rank_ts + (1-w)*rank_au; sort ascending
    - Compute precision@10 for each w.
- Aggregate p@10 across campaigns; pick w maximizing mean p@10.

Output:
- AELP2/reports/auction_ensemble_summary.json
"""
import os, json, math
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
CREATIVE_DIR = ROOT / 'AELP2' / 'reports' / 'creative'
OUT = ROOT / 'AELP2' / 'reports' / 'auction_ensemble_summary.json'

import sys
sys.path.insert(0, str(ROOT / 'AELP2' / 'tools'))
from backtest_auction_accuracy import (
    derive_rates, predict_cac_auction, fetch_metrics_for_campaign,
    load_cpc_model, extract_features_for_ad, precision_at_k
)
from auction_gym_integration import AuctionGymWrapper


def rank_map(sorted_ids: List[str]) -> Dict[str, int]:
    return {cid: i+1 for i, cid in enumerate(sorted_ids)}


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT','aura-thrive-platform')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET','gaelp_training')
    bq = bigquery.Client(project=project)
    auction = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})
    cpc_model, cpc_keys = load_cpc_model()

    weight_grid = [i/20.0 for i in range(0, 21)]  # 0.0 .. 1.0 step 0.05
    p10_ts, p10_au = [], []
    p10_w = {w: [] for w in weight_grid}
    details = []

    for f in sorted(CREATIVE_DIR.glob('*.json')):
        d = json.loads(f.read_text())
        cid = str(d.get('campaign_id') or f.stem)
        items = d.get('items') or []
        if len(items) < 3:
            continue
        labels = {str(it['creative_id']): int(it.get('actual_label') or 0) for it in items}
        sim_scores = {str(it['creative_id']): float(it.get('sim_score') or 0.0) for it in items}

        # TS ranking: high sim_score first
        ts_ranked = [ad for ad,_ in sorted(sim_scores.items(), key=lambda t: -t[1])]
        ts_ranks = rank_map(ts_ranked)

        # Auction CACs
        metrics = fetch_metrics_for_campaign(bq, project, dataset, cid)
        auc_cac: Dict[str, float] = {}
        for ad_id in labels.keys():
            m = metrics.get(str(ad_id))
            if not m or m.get('imps',0.0) < 500 or m.get('clicks',0.0) < 20:
                continue
            ctr, cvr = derive_rates(m)
            cpc = None; cac = None
            if cpc_model is not None and cpc_keys:
                try:
                    X = extract_features_for_ad(str(ad_id), cpc_keys)
                    cpc_hat = float(cpc_model.predict(X)[0])
                    if math.isfinite(cpc_hat) and cpc_hat>0:
                        cpc = cpc_hat
                        cac = cpc_hat / max(cvr, 1e-6)
                except Exception:
                    pass
            if cpc is None:
                _cpc, _cac = predict_cac_auction(auction, ctr, cvr, rounds=600)
                cpc, cac = _cpc, _cac
            if math.isfinite(cac):
                auc_cac[str(ad_id)] = float(cac)
        if not auc_cac:
            continue
        au_ranked = [ad for ad,_ in sorted(auc_cac.items(), key=lambda t: t[1])]
        au_ranks = rank_map(au_ranked)

        # Evaluate
        p10_ts.append(precision_at_k(ts_ranked, labels, 10))
        p10_au.append(precision_at_k(au_ranked, labels, 10))
        for w in weight_grid:
            scores = {}
            for ad in labels.keys():
                rt = ts_ranks.get(ad, len(labels)+1)
                ra = au_ranks.get(ad, len(labels)+1)
                scores[ad] = w*rt + (1-w)*ra
            ens_ranked = [ad for ad,_ in sorted(scores.items(), key=lambda t: t[1])]
            p10_w[w].append(precision_at_k(ens_ranked, labels, 10))
        details.append({'campaign_id': cid, 'n': len(items)})

    # Aggregate
    mean_ts = float(np.nanmean(p10_ts)) if p10_ts else None
    mean_au = float(np.nanmean(p10_au)) if p10_au else None
    best_w, best_p = None, -1.0
    for w, arr in p10_w.items():
        if not arr:
            continue
        m = float(np.nanmean(arr))
        if m > best_p:
            best_p, best_w = m, w
    out = {
        'mean_p10_ts': round(mean_ts,3) if mean_ts is not None else None,
        'mean_p10_auction': round(mean_au,3) if mean_au is not None else None,
        'best_weight': best_w,
        'mean_p10_ensemble': round(best_p,3) if best_p is not None else None,
        'weights_grid': {str(w): (round(float(np.nanmean(v)),3) if v else None) for w,v in p10_w.items()},
        'n_campaigns': len(details),
        'note': 'weights interpolate ranks: score = w*rank_ts + (1-w)*rank_auction'
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
