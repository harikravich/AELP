#!/usr/bin/env python3
from __future__ import annotations
"""
Train a per-ad CPC regression model from local weekly creatives + creative objects.

Data sources (local, no BQ required):
- AELP2/reports/weekly_creatives/*.json → labels: CPC = test_spend / test_clicks
- AELP2/reports/creative_objects/<creative_id>.json → features: is_video, formats, positions, CTA, etc.

Outputs:
- AELP2/reports/models/cpc_model.joblib (sklearn GradientBoostingRegressor)
- AELP2/reports/models/cpc_model_info.json (feature schema + metrics)

Usage:
  python3 AELP2/tools/train_cpc_model.py

Notes:
- Filters out rows with < 50 clicks or CPC outside [0.05, 20.0] to remove noise/outliers.
- Uses campaign-grouped split for validation (leave-one-campaign-out like) to reduce leakage.
"""
import json, math, os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

def _require(mod: str):
    try:
        return __import__(mod)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])
        return __import__(mod)

sklearn = _require('sklearn')
joblib = _require('joblib')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
WK = ROOT / 'AELP2' / 'reports' / 'weekly_creatives'
OBJ = ROOT / 'AELP2' / 'reports' / 'creative_objects'
OUTDIR = ROOT / 'AELP2' / 'reports' / 'models'
OUTDIR.mkdir(parents=True, exist_ok=True)


def extract_features(creative_id: str) -> Dict[str, float]:
    """Build a simple feature vector from creative_objects JSON (sparse-safe)."""
    p = OBJ / f"{creative_id}.json"
    feats: Dict[str, float] = {}
    def setf(k, v):
        feats[k] = float(v)
    def inc(k, v=1.0):
        feats[k] = feats.get(k, 0.0) + float(v)
    if not p.exists():
        # minimal placeholder features
        setf('bias', 1.0)
        return feats
    try:
        d = json.loads(p.read_text())
    except Exception:
        setf('bias', 1.0)
        return feats
    setf('bias', 1.0)
    # Basic signals
    cr = (d.get('creative') or {})
    afs = (cr.get('asset_feed_spec') or {})
    videos = afs.get('videos') or []
    setf('is_video', 1.0 if videos else 0.0)
    # Count text assets
    titles = afs.get('titles') or []
    bodies = afs.get('bodies') or []
    setf('n_titles', len(titles))
    setf('n_bodies', len(bodies))
    # CTA types
    ctas = afs.get('call_to_action_types') or []
    for c in ctas:
        inc(f"cta_{str(c).lower()}")
    # Positions/platforms from customization rules
    rules = afs.get('asset_customization_rules') or []
    pos_keys = set()
    plat_keys = set()
    for r in rules:
        cs = (r.get('customization_spec') or {})
        for pos in (cs.get('facebook_positions') or []):
            pos_keys.add(f"fbpos_{pos}")
        for pos in (cs.get('instagram_positions') or []):
            pos_keys.add(f"igpos_{pos}")
        for pos in (cs.get('audience_network_positions') or []):
            pos_keys.add(f"anpos_{pos}")
        for plat in (cs.get('publisher_platforms') or []):
            plat_keys.add(f"plat_{plat}")
    for k in sorted(pos_keys):
        inc(k)
    for k in sorted(plat_keys):
        inc(k)
    # Optimization type
    opt = cr.get('asset_feed_spec', {}).get('optimization_type') or cr.get('optimization_type')
    if opt:
        inc(f"opt_{str(opt).lower()}")
    return feats


def collect_training_rows() -> Tuple[List[Dict[str,float]], List[float], List[str]]:
    X_rows: List[Dict[str,float]] = []
    y: List[float] = []
    groups: List[str] = []  # campaign ids for grouped split
    for wf in sorted(WK.glob('*.json')):
        try:
            d = json.loads(wf.read_text())
        except Exception:
            continue
        cid = str(d.get('campaign_id') or wf.name.split('_')[0])
        for it in d.get('items') or []:
            clicks = float(it.get('test_clicks') or 0.0)
            spend = float(it.get('test_spend') or 0.0)
            if clicks < 50.0 or spend <= 0:
                continue
            cpc = spend / clicks
            if not (0.05 <= cpc <= 20.0):
                continue
            feats = extract_features(str(it.get('creative_id')))
            X_rows.append(feats)
            y.append(cpc)
            groups.append(cid)
    return X_rows, y, groups


def vectorize_dicts(rows: List[Dict[str,float]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for r in rows for k in r.keys()})
    mat = np.zeros((len(rows), len(keys)), dtype=np.float32)
    key_index = {k:i for i,k in enumerate(keys)}
    for i, r in enumerate(rows):
        for k, v in r.items():
            mat[i, key_index[k]] = float(v)
    return mat, keys


def main():
    X_rows, y, groups = collect_training_rows()
    if len(y) < 200:
        info = {'status':'insufficient_data','n':len(y)}
        (OUTDIR / 'cpc_model_info.json').write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2)); return
    X, keys = vectorize_dicts(X_rows)
    y_arr = np.array(y, dtype=np.float32)
    # Simple grouped split: last 20% campaigns as validation
    camp_order = []
    seen = set()
    for g in groups:
        if g not in seen:
            seen.add(g); camp_order.append(g)
    split_idx = int(len(camp_order)*0.8)
    train_camps = set(camp_order[:split_idx])
    mask_train = np.array([g in train_camps for g in groups])
    Xtr, ytr = X[mask_train], y_arr[mask_train]
    Xte, yte = X[~mask_train], y_arr[~mask_train]
    # Model
    model = GradientBoostingRegressor(random_state=42, loss='squared_error', n_estimators=200, max_depth=3, learning_rate=0.05)
    model.fit(Xtr, ytr)
    pred_tr = model.predict(Xtr); pred_te = model.predict(Xte)
    mae_tr = float(mean_absolute_error(ytr, pred_tr))
    rmse_tr = float(mean_squared_error(ytr, pred_tr)) ** 0.5
    mae_te = float(mean_absolute_error(yte, pred_te))
    rmse_te = float(mean_squared_error(yte, pred_te)) ** 0.5
    joblib.dump({'model': model, 'features': keys}, OUTDIR / 'cpc_model.joblib')
    info = {
        'status': 'ok',
        'n': int(len(y)),
        'n_features': len(keys),
        'mae_train': mae_tr,
        'rmse_train': rmse_tr,
        'mae_val': mae_te,
        'rmse_val': rmse_te,
        'features': keys[:50] + ([f"... +{len(keys)-50} more"] if len(keys)>50 else [])
    }
    (OUTDIR / 'cpc_model_info.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))

if __name__ == '__main__':
    main()
