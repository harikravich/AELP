#!/usr/bin/env python3
from __future__ import annotations
"""
Score local final videos with the Meta-only new-ad ranker (light features).
Features are proxied from filename since we don't have copy/meta for new assets.

Outputs: AELP2/reports/new_ad_scores.json
"""
import json, joblib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT/'AELP2'/'outputs'/'finals'
MOD = ROOT/'AELP2'/'models'/'new_ad_ranker'
OUT = ROOT/'AELP2'/'reports'/'new_ad_scores.json'

def load_model():
    clf=joblib.load(MOD/'model.pkl')
    iso=joblib.load(MOD/'calib.pkl')
    fmap=json.loads((MOD/'feature_map.json').read_text())
    ref=json.loads((MOD/'feature_ref.json').read_text())['ref']
    conf=json.loads((MOD/'conformal.json').read_text())
    return clf, iso, fmap, np.asarray(ref, float), conf

def proxy_features(name: str, fmap: dict) -> np.ndarray:
    # crude proxies; replace with real copy/meta when available
    feats={
        'title_len': 24,
        'body_len': 100,
        'has_link': 1,
        'p_fb_story': 1,
        'p_fb_reels': 1,
        'p_ig_story': 1,
        'p_ig_reels': 1,
        'name_len': len(name)
    }
    return np.asarray([float(feats.get(k,0.0)) for k in fmap.keys()], float)

def main():
    clf, iso, fmap, ref, conf = load_model()
    rows=[]
    for mp4 in sorted(FIN.glob('*.mp4')):
        vf = proxy_features(mp4.name, fmap)
        x = vf - ref
        p = float(clf.predict_proba([x])[0,1])
        p_cal = float(iso.transform([p])[0])
        lcb = max(0.0, p_cal - float(conf.get('qhat', 0.2)))
        rows.append({'file': mp4.name, 'p_win': round(p_cal,4), 'lcb': round(lcb,4)})
    OUT.write_text(json.dumps({'items': rows}, indent=2))
    print(json.dumps({'scored': len(rows), 'out': str(OUT)}, indent=2))

if __name__=='__main__':
    main()

