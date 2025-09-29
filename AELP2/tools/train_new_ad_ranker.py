#!/usr/bin/env python3
from __future__ import annotations
"""
Train a lightweight pairwise ranker on weekly pairs (feature diffs).
Model: LogisticRegression on X (diff vector) -> P(win vs baseline)
Calibration: Isotonic on held-out fold; Conformal: absolute residual quantile.

Artifacts saved to AELP2/models/new_ad_ranker/
  - model.pkl (sklearn pipeline)
  - calib.pkl (IsotonicRegression)
  - conformal.json (qhat, alpha)
  - feature_map.json (copied from reports)
  - train_report.json (metrics)
"""
import json, joblib, random
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[2]
PAIRS = ROOT/ 'AELP2' / 'reports' / 'new_ranker' / 'pairs.jsonl'
FMAP  = ROOT/ 'AELP2' / 'reports' / 'new_ranker' / 'feature_map.json'
OUTD  = ROOT/ 'AELP2' / 'models' / 'new_ad_ranker'

def load_pairs():
    X=[]; y=[]
    with PAIRS.open() as f:
        for line in f:
            d=json.loads(line)
            X.append(d['x']); y.append(d['y'])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

def split_idx(n, seed=0):
    idx=list(range(n)); random.Random(seed).shuffle(idx)
    k=int(n*0.8)
    return np.array(idx[:k]), np.array(idx[k:])

def main():
    X,y = load_pairs()
    if len(X)==0:
        print(json.dumps({'error':'no pairs'})); return
    OUTD.mkdir(parents=True, exist_ok=True)
    i_tr,i_va = split_idx(len(X))
    Xtr,Xva = X[i_tr], X[i_va]; ytr,yva = y[i_tr], y[i_va]
    clf=LogisticRegression(max_iter=200)
    clf.fit(Xtr,ytr)
    p_tr = clf.predict_proba(Xtr)[:,1]
    p_va = clf.predict_proba(Xva)[:,1]
    # Isotonic calibration on val
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(p_va, yva)
    p_cal = iso.transform(p_va)
    # Conformal qhat (absolute residuals at 1-alpha)
    alpha=0.1
    resid = np.abs(yva - p_cal)
    qhat = float(np.quantile(resid, 1-alpha))
    joblib.dump(clf, OUTD/'model.pkl')
    joblib.dump(iso, OUTD/'calib.pkl')
    # copy fmap
    fmap=json.loads(FMAP.read_text()) if FMAP.exists() else {}
    (OUTD/'feature_map.json').write_text(json.dumps(fmap, indent=2))
    (OUTD/'conformal.json').write_text(json.dumps({'alpha': alpha, 'qhat': qhat}, indent=2))
    # feature reference (mean of original features; used as pseudo-baseline)
    # Load full features table to compute a reference vector
    from pathlib import Path as P
    FEATJL = P('AELP2/reports/creative_features/creative_features.jsonl')
    ref=[0.0]*X.shape[1]
    try:
        import json as jsonlib
        s=0
        with FEATJL.open() as f:
            for line in f:
                d=jsonlib.loads(line)
                vec=[float(d.get(k) or 0.0) for k in jsonlib.loads((OUTD/'feature_map.json').read_text()).keys()]
                ref=[a+b for a,b in zip(ref, vec)]; s+=1
        if s>0:
            ref=[v/s for v in ref]
    except Exception:
        pass
    (OUTD/'feature_ref.json').write_text(json.dumps({'ref': ref}, indent=2))
    report={
        'n': int(len(X)),
        'auc_tr': float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else None,
        'auc_va': float(roc_auc_score(yva, p_va)) if len(np.unique(yva))>1 else None,
        'acc_va': float(accuracy_score(yva, (p_va>=0.5).astype(int))),
        'acc_va_cal': float(accuracy_score(yva, (p_cal>=0.5).astype(int))),
        'qhat': qhat
    }
    (OUTD/'train_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'saved': str(OUTD), **report}, indent=2))

if __name__=='__main__':
    main()
