#!/usr/bin/env python3
from __future__ import annotations
"""
Offline RL stub (bandit/IQL-lite): trains a logistic policy to predict P(win) from features and applies a conservative penalty.
Outputs: AELP2/reports/offline_policy.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
ENR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'

def featurize(it: dict):
    dna = (it.get('dna_real') or {})
    bow = dna.get('bow64') or dna.get('bow16') or []
    bow = bow if isinstance(bow, list) else []
    x = [float(dna.get('title_wc') or 0.0), float(dna.get('avg_word_len') or 0.0), float(dna.get('has_cta_button') or 0.0)]
    x += bow[:64] + [0.0]*(64-len(bow[:64]))
    return np.array(x, dtype=float)

def main():
    X=[]; y=[]; ids=[]
    for f in sorted(ENR.glob('*.json')):
        d=json.loads(f.read_text()); items=d.get('items') or []
        for it in items:
            X.append(featurize(it)); y.append(int(it.get('actual_label',0))); ids.append(it.get('creative_id'))
    if not X:
        print('no data'); return
    X=np.vstack(X); y=np.array(y)
    # Conservative training: stronger regularization and class_weight to avoid overconfidence
    clf=LogisticRegression(max_iter=1000, C=0.2, class_weight='balanced')
    clf.fit(X,y)
    p=clf.predict_proba(X)[:,1]
    # Conservative value: subtract a margin proportional to uncertainty
    margin=0.1
    score = np.maximum(0.0, p - margin)
    ranked = np.argsort(-score)[:50]
    out={'policy': {'type': 'logistic_conservative', 'margin': margin, 'C': 0.2},
         'top50': [{'creative_id': ids[i], 'p_conservative': float(score[i])} for i in ranked]}
    OUT.joinpath('offline_policy.json').write_text(json.dumps(out, indent=2))
    print('AELP2/reports/offline_policy.json')

if __name__=='__main__':
    main()

