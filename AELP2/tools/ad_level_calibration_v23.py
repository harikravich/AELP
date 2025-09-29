#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
CRE_DIR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
if not CRE_DIR.exists():
    raise SystemExit('creative_enriched not found; run enrich_creatives_with_objects.py')
OUT = ROOT / 'AELP2' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

def top_placement_keys(files: List[Path], k: int = 8) -> List[str]:
    counts = {}
    for f in files:
        data = json.loads(f.read_text())
        for it in (data.get('items') or []):
            for key in (it.get('placement_mix') or {}).keys():
                counts[key] = counts.get(key, 0) + 1
    keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return keys[:k]

def one_hot_ratio(tag: str) -> List[float]:
    return [1.0 if tag=='9x16' else 0.0, 1.0 if tag=='1x1-4x5' else 0.0]

def featurize(it: dict, plac_keys: List[str]) -> List[float]:
    x: List[float] = []
    mix = it.get('placement_mix') or {}
    for k in plac_keys:
        x.append(float(mix.get(k, 0.0)))
    dna_real = it.get('dna_real') or {}
    x += [float(dna_real.get('title_wc') or 0.0), float(dna_real.get('avg_word_len') or 0.0), float(dna_real.get('has_cta_button') or 0.0)]
    x += one_hot_ratio(dna_real.get('ratio_tag') or '')
    age = float(dna_real.get('age_days') or 0.0)
    x += [age, age*age/10000.0]  # simple quadratic
    # CPC and fatigue features
    x += [float(it.get('test_cpc') or 0.0)]
    fs = it.get('fatigue_slope'); x += [float(fs) if fs is not None else 0.0]
    # baseline sim_score as a feature (last)
    x += [float(it.get('sim_score') or 0.0)]
    return x

def evaluate(files: List[Path]):
    plac = top_placement_keys(files)
    metrics = []
    Y_all, P_all = [], []
    for hold in files:
        Xtr, ytr = [], []
        Xte, yte, ids = [], [], []
        for f in files:
            data = json.loads(f.read_text())
            items = data.get('items') or []
            for it in items:
                y = int(it.get('actual_label', 0))
                x = featurize(it, plac)
                if f == hold:
                    Xte.append(x); yte.append(y); ids.append(it.get('creative_id'))
                else:
                    Xtr.append(x); ytr.append(y)
        if not Xte or sum(ytr) == 0:
            continue
        Xtr = np.array(Xtr); ytr = np.array(ytr)
        Xte = np.array(Xte); yte = np.array(yte)
        clf = LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced')
        clf.fit(Xtr, ytr)
        p_raw = clf.predict_proba(Xtr)[:,1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_raw, ytr)
        p_te = iso.transform(clf.predict_proba(Xte)[:,1])
        order = np.argsort(-p_te)
        k5 = order[:5]; k10 = order[:10]
        p5 = float(np.mean([yte[i] for i in k5])) if len(order)>=5 else float('nan')
        p10 = float(np.mean([yte[i] for i in k10])) if len(order)>=10 else float('nan')
        wins=tot=0
        for i in range(len(order)):
            for j in range(i+1,len(order)):
                yi, yj = yte[order[i]], yte[order[j]]
                if yi==yj: continue; tot+=1; wins += 1 if yi>yj else 0
        pw = wins/tot if tot else float('nan')
        metrics.append({'campaign_file': hold.name, 'precision_at_5': p5, 'precision_at_10': p10, 'pairwise_win_rate': pw})
        Y_all.extend(list(yte)); P_all.extend(list(p_te))
    def avg(v):
        v=[x for x in v if x==x]; return float(np.mean(v)) if v else None
    summary = {'precision_at_5': avg([m['precision_at_5'] for m in metrics]),
               'precision_at_10': avg([m['precision_at_10'] for m in metrics]),
               'pairwise_win_rate': avg([m['pairwise_win_rate'] for m in metrics]),
               'n_campaigns': len(metrics), 'n_samples': len(Y_all)}
    OUT.joinpath('ad_level_accuracy_v23.json').write_text(json.dumps(summary, indent=2))
    OUT.joinpath('ad_level_accuracy_v23_details.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(summary, indent=2))

def main():
    files = sorted(CRE_DIR.glob('*.json'))
    if not files:
        print('No enriched creative files found.')
        return
    evaluate(files)

if __name__ == '__main__':
    main()
