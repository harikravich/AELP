#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import os
CRE_DIR = Path(os.getenv('AELP2_CREATIVE_DIR') or (ROOT / 'AELP2' / 'reports' / 'creative_enriched'))
OUT = ROOT / 'AELP2' / 'reports'

def load_scores(selector: str = 'v24'):
    # Load per-ad final scores from selector or recompute simple proxies
    # For now, re-run the same scoring as v24 inline (utility + sim mix)
    from ad_level_ranker_v24 import top_placement_keys, featurize, normalize_sim, build_pairs
    files = sorted(CRE_DIR.glob('*.json'))
    plac = top_placement_keys(files)
    # Build a single global model using all campaigns (approximate)
    feats=[]; labels=[]; sims=[]
    for f in files:
        d=json.loads(f.read_text())
        for it in (d.get('items') or []):
            feats.append(featurize(it, plac)); labels.append(1 if it.get('actual_label') else 0); sims.append(float(it.get('sim_score') or 0.0))
    import numpy as np
    X=np.vstack(feats); y=np.array(labels); S=np.array(sims)
    from sklearn.linear_model import LogisticRegression
    PX, Py = build_pairs(X, y)
    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    if len(Py)==0:
        util = np.zeros(len(y))
    else:
        clf.fit(PX, Py)
        util = (X @ clf.coef_.reshape(-1,1)).squeeze()
    # Normalize
    umin, umax = float(util.min()), float(util.max())
    util_n = (util-umin)/(umax-umin) if umax>umin else np.full_like(util, 0.5)
    sim_n = np.array(normalize_sim(list(S)))
    w=0.7
    final = w*sim_n + (1.0-w)*util_n
    return final, y

def main():
    scores, labels = load_scores('v24')
    # Coverage curve by threshold
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    n=len(scores)
    ks=[1,3,5,10,20]
    curves={}
    for k in ks:
        if n==0: curves[str(k)]={}; continue
        topk_labels = sorted_labels[:k]
        cov = float(np.mean(topk_labels)) if len(topk_labels) else 0.0
        curves[str(k)]={'precision': cov}
    out={'selector':'v24','curves':curves}
    OUT.joinpath('conformal_topk_v24.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__=='__main__':
    main()
