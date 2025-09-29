#!/usr/bin/env python3
from __future__ import annotations
"""
Score all creative_objects (including vendor-imported) with the trained new-ad ranker.

Steps:
  1) Ensure features are built (build_features_from_creative_objects.py)
  2) Load model, calibrator, feature map, and reference vector
  3) Score each creative_id and write vendor_scores.json

Outputs: AELP2/reports/vendor_scores.json
"""
import json, joblib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATJL = ROOT / 'AELP2' / 'reports' / 'creative_features' / 'creative_features.jsonl'
MOD = ROOT / 'AELP2' / 'models' / 'new_ad_ranker'
OUT = ROOT / 'AELP2' / 'reports' / 'vendor_scores.json'


def load_model():
    clf = joblib.load(MOD / 'model.pkl')
    iso = joblib.load(MOD / 'calib.pkl')
    fmap = json.loads((MOD / 'feature_map.json').read_text())
    ref = json.loads((MOD / 'feature_ref.json').read_text())['ref']
    conf = json.loads((MOD / 'conformal.json').read_text())
    return clf, iso, list(fmap.keys()), np.asarray(ref, float), float(conf.get('qhat', 0.2))


def load_features(keys: list[str]):
    feats = {}
    if not FEATJL.exists():
        return feats
    with FEATJL.open() as f:
        for line in f:
            d = json.loads(line)
            cid = d.get('creative_id')
            x = [float(d.get(k, 0.0)) for k in keys]
            feats[cid] = np.asarray(x, float)
    return feats


def main():
    clf, iso, keys, ref, qhat = load_model()
    feats = load_features(keys)
    rows = []
    for cid, vf in feats.items():
        x = vf - ref
        p = float(clf.predict_proba([x])[0, 1])
        p_cal = float(iso.transform([p])[0])
        lcb = max(0.0, p_cal - qhat)
        rows.append({'creative_id': cid, 'p_win': round(p_cal, 4), 'lcb': round(lcb, 4)})
    rows.sort(key=lambda r: (-r['p_win'], r['creative_id']))
    OUT.write_text(json.dumps({'items': rows, 'count': len(rows)}, indent=2))
    print(json.dumps({'scored': len(rows), 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()

