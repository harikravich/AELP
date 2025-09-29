#!/usr/bin/env python3
from __future__ import annotations
"""
Train an XGBoost pairwise ranker to predict creative order within campaigns.

Features per ad:
- TS score (sim_score from creative/*.json)
- Auction CAC prediction (from backtest path using CPC model if available)
- Evidence: CTR, CVR, clicks, imps (from BQ)
- Creative DNA: is_video, n_titles, n_bodies, CTA/positions/platforms (from creative_objects)

Labels:
- actual_label from creative/*.json (1 = positive, 0 = negative)

Groups:
- One group per campaign_id.

Outputs:
- AELP2/reports/models/pairwise_ranker.json (xgboost Booster saved via xgb_model.json)
- AELP2/reports/models/pairwise_ranker_info.json (metrics + feature names)
"""
import os, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

def _req(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])
        return __import__(mod)

xgb = _req('xgboost')
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parents[2]
CREATIVE_DIR = ROOT / 'AELP2' / 'reports' / 'creative'
OBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUTDIR = ROOT / 'AELP2' / 'reports' / 'models'
OUTDIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT / 'AELP2' / 'tools'))
from backtest_auction_accuracy import (
    derive_rates, predict_cac_auction, fetch_metrics_for_campaign,
    load_cpc_model, extract_features_for_ad
)


def feat_creative_dna(creative_id: str) -> Dict[str, float]:
    p = OBJ / f"{creative_id}.json"
    feats: Dict[str,float] = {'bias': 1.0}
    if not p.exists():
        return feats
    try:
        d=json.loads(p.read_text())
    except Exception:
        return feats
    cr = (d.get('creative') or {})
    afs = (cr.get('asset_feed_spec') or {})
    videos = afs.get('videos') or []
    feats['is_video']=1.0 if videos else 0.0
    feats['n_titles']=float(len(afs.get('titles') or []))
    feats['n_bodies']=float(len(afs.get('bodies') or []))
    for cta in (afs.get('call_to_action_types') or []):
        feats[f"cta_{str(cta).lower()}"]=1.0
    rules = afs.get('asset_customization_rules') or []
    for r in rules[:2]:  # limit
        cs = (r.get('customization_spec') or {})
        for pos in (cs.get('facebook_positions') or []):
            feats[f"fbpos_{pos}"]=1.0
        for pos in (cs.get('instagram_positions') or []):
            feats[f"igpos_{pos}"]=1.0
        for pos in (cs.get('audience_network_positions') or []):
            feats[f"anpos_{pos}"]=1.0
        for plat in (cs.get('publisher_platforms') or []):
            feats[f"plat_{plat}"]=1.0
    return feats


def vectorize(rows: List[Dict[str,float]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for r in rows for k in r.keys()})
    X = np.zeros((len(rows), len(keys)), dtype=np.float32)
    idx = {k:i for i,k in enumerate(keys)}
    for i, r in enumerate(rows):
        for k,v in r.items():
            X[i, idx[k]] = float(v)
    return X, keys


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT','aura-thrive-platform')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET','gaelp_training')
    bq = bigquery.Client(project=project)
    cpc_model, cpc_keys = load_cpc_model()

    X_rows: List[Dict[str,float]] = []
    y: List[int] = []
    group_ptrs: List[int] = []
    group_sizes: List[int] = []
    sim_scores: List[float] = []

    for f in sorted(CREATIVE_DIR.glob('*.json')):
        d=json.loads(f.read_text()); cid=str(d.get('campaign_id') or f.stem)
        items=d.get('items') or []
        if len(items) < 3:
            continue
        metrics = fetch_metrics_for_campaign(bq, project, dataset, cid)
        start_index = len(y)
        for it in items:
            ad=str(it.get('creative_id'))
            m=metrics.get(ad)
            if not m or m.get('imps',0.0) < 500 or m.get('clicks',0.0) < 20:
                continue
            ctr, cvr = derive_rates(m)
            # Auction CAC (with CPC model fallback)
            if cpc_model is not None and cpc_keys:
                try:
                    Xc = extract_features_for_ad(ad, cpc_keys)
                    cpc_hat = float(cpc_model.predict(Xc)[0])
                    cac_hat = cpc_hat / max(cvr, 1e-6)
                except Exception:
                    _, cac_hat = predict_cac_auction(None, ctr, cvr, rounds=400)  # type: ignore
            else:
                _, cac_hat = predict_cac_auction(None, ctr, cvr, rounds=400)  # type: ignore
            dna = feat_creative_dna(ad)
            row = {
                **dna,
                'sim_score': float(it.get('sim_score') or 0.0),
                'ctr': float(ctr),
                'cvr': float(cvr),
                'clicks': float(m.get('clicks') or 0.0),
                'imps': float(m.get('imps') or 0.0),
                'auc_cac': float(cac_hat),
            }
            X_rows.append(row)
            y.append(int(it.get('actual_label') or 0))
            sim_scores.append(float(it.get('sim_score') or 0.0))
        size = len(y) - start_index
        if size > 0:
            group_ptrs.append(start_index)
            group_sizes.append(size)

    if not y or sum(group_sizes)==0:
        OUTDIR.joinpath('pairwise_ranker_info.json').write_text(json.dumps({'status':'insufficient_data','n':len(y)}, indent=2))
        print(json.dumps({'status':'insufficient_data','n':len(y)}, indent=2)); return

    X, keys = vectorize(X_rows)
    dtrain = xgb.DMatrix(X, label=np.array(y, dtype=np.float32))
    dtrain.set_group(group_sizes)
    params = {
        'objective': 'rank:pairwise',
        'eta': 0.08,
        'max_depth': 5,
        'min_child_weight': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'ndcg@10',
        'seed': 42,
    }
    bst = xgb.train(params, dtrain, num_boost_round=200)
    # Save
    model_path = OUTDIR / 'pairwise_ranker.json'
    bst.save_model(str(model_path))
    info = {
        'status': 'ok',
        'n_rows': int(X.shape[0]),
        'n_features': len(keys),
        'groups': len(group_sizes),
        'feature_names': keys[:60] + ([f"... +{len(keys)-60} more"] if len(keys)>60 else [])
    }
    OUTDIR.joinpath('pairwise_ranker_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()

