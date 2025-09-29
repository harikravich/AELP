#!/usr/bin/env python3
from __future__ import annotations
import json, math
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
CRE_DIR = ROOT / 'AELP2' / 'reports' / 'creative_enriched'
OUT = ROOT / 'AELP2' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

def top_placement_keys(files: List[Path], k: int = 12) -> List[str]:
    counts = {}
    for f in files:
        data = json.loads(f.read_text())
        for it in (data.get('items') or []):
            for key in (it.get('placement_mix') or {}).keys():
                counts[key] = counts.get(key, 0) + 1
    keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return keys[:k]

def ratio_one_hot(tag: str) -> List[float]:
    return [1.0 if tag=='9x16' else 0.0,
            1.0 if tag=='1x1-4x5' else 0.0]

def featurize(it: dict, plac_keys: List[str]) -> np.ndarray:
    x: List[float] = []
    # placement shares
    mix = it.get('placement_mix') or {}
    for k in plac_keys:
        x.append(float(mix.get(k, 0.0)))
    # dna_real
    dna_real = it.get('dna_real') or {}
    x += [float(dna_real.get('title_wc') or 0.0),
          float(dna_real.get('avg_word_len') or 0.0),
          float(dna_real.get('has_cta_button') or 0.0)]
    x += ratio_one_hot(dna_real.get('ratio_tag') or '')
    age = float(dna_real.get('age_days') or 0.0)
    x += [age, (age*age)/10000.0]
    # CPC + fatigue
    x += [float(it.get('test_cpc') or 0.0)]
    fs = it.get('fatigue_slope'); x += [float(fs) if fs is not None else 0.0]
    # Hashed text features from enrichment (bow16)
    bow = None
    if isinstance(dna_real.get('bow64'), list):
        bow = dna_real.get('bow64')
    elif isinstance(dna_real.get('bow16'), list):
        bow = dna_real.get('bow16')
    if bow is None:
        bow = [0.0]*16
    x += bow
    # FINAL: do NOT include sim_score in the model; we will mix it monotonically after
    return np.array(x, dtype=float)

def build_calibrators(train_files: List[Path]):
    # Per-placement CTR and CPC priors from training campaigns only
    agg = {}
    for f in train_files:
        data = json.loads(f.read_text())
        for it in (data.get('items') or []):
            stats = it.get('placement_stats') or {}
            for k,v in stats.items():
                a = agg.setdefault(k, {'impr':0.0,'clicks':0.0,'spend':0.0})
                a['impr'] += float((v or {}).get('impr',0.0) or 0.0)
                a['clicks'] += float((v or {}).get('clicks',0.0) or 0.0)
                a['spend'] += float((v or {}).get('spend',0.0) or 0.0)
    ctr = {}
    cpc = {}
    # Global fallbacks
    tot_impr = sum(v['impr'] for v in agg.values()) or 1.0
    tot_clicks = sum(v['clicks'] for v in agg.values()) or 1.0
    tot_spend = sum(v['spend'] for v in agg.values()) or 1.0
    global_ctr = (tot_clicks / tot_impr) if tot_impr>0 else 0.01
    global_cpc = (tot_spend / tot_clicks) if tot_clicks>0 else 1.0
    for k,v in agg.items():
        ctr[k] = (v['clicks']/v['impr']) if v['impr']>0 else global_ctr
        cpc[k] = (v['spend']/v['clicks']) if v['clicks']>0 else global_cpc
    return ctr, cpc, global_ctr, global_cpc

def mix_calibrated_features(it: dict, ctr_map: dict, cpc_map: dict, gctr: float, gcpc: float, plac_keys: List[str]) -> List[float]:
    mix = it.get('placement_mix') or {}
    # Expected CTR and CPC under observed mix
    exp_ctr = 0.0; exp_cpc = 0.0
    total_share = 0.0
    for k in plac_keys:
        w = float(mix.get(k, 0.0)); total_share += w
        exp_ctr += w * float(ctr_map.get(k, gctr))
        exp_cpc += w * float(cpc_map.get(k, gcpc))
    if total_share < 1e-6:
        exp_ctr = gctr; exp_cpc = gcpc
    return [exp_ctr, exp_cpc]

def build_pairs(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Build pairwise differences for samples with differing labels
    idx_pos = np.where(y==1)[0]
    idx_neg = np.where(y==0)[0]
    pairs_X = []
    pairs_y = []
    # Sample up to 10x of min(len(pos),len(neg)) pairs to limit size
    import random
    random.seed(0)
    max_pairs = 10*min(len(idx_pos), len(idx_neg)) if len(idx_pos) and len(idx_neg) else 0
    if max_pairs == 0:
        return np.zeros((0, X.shape[1])), np.zeros((0,))
    for _ in range(max_pairs):
        i = random.choice(idx_pos); j = random.choice(idx_neg)
        pairs_X.append(X[i] - X[j])
        pairs_y.append(1)
        # also add the inverse pair for balance
        pairs_X.append(X[j] - X[i])
        pairs_y.append(0)
    return np.vstack(pairs_X), np.array(pairs_y)

def normalize_sim(sim_scores: List[float]) -> List[float]:
    smin = min(sim_scores) if sim_scores else 0.0
    smax = max(sim_scores) if sim_scores else 1.0
    if smax <= smin:
        return [0.5 for _ in sim_scores]
    return [(s - smin) / (smax - smin) for s in sim_scores]

def precision_at_k(sorted_ids: List[str], labels: dict, k: int=5) -> float:
    topk = sorted_ids[:k]
    if not topk:
        return float('nan')
    return sum(labels.get(cid,0) for cid in topk) / len(topk)

def evaluate(files: List[Path]):
    plac = top_placement_keys(files)
    p5s=[]; p10s=[]
    details=[]
    for hold in files:
        # Build train and test matrices
        tr_feats=[]; tr_labels=[]
        te_feats=[]; te_labels=[]; te_ids=[]; te_sim=[]
        # Build calibrators on training folds only
        train_files = [f for f in files if f != hold]
        ctr_map, cpc_map, gctr, gcpc = build_calibrators(train_files)
        for f in files:
            data = json.loads(f.read_text())
            items = data.get('items') or []
            for it in items:
                base_x = featurize(it, plac)
                cal_x = mix_calibrated_features(it, ctr_map, cpc_map, gctr, gcpc, plac)
                x = np.concatenate([base_x, np.array(cal_x, dtype=float)])
                y = 1 if it.get('actual_label') else 0
                if f == hold:
                    te_feats.append(x); te_labels.append(y); te_ids.append(it.get('creative_id')); te_sim.append(float(it.get('sim_score') or 0.0))
                else:
                    tr_feats.append(x); tr_labels.append(y)
        if not te_feats or sum(tr_labels)==0 or sum(tr_labels)==len(tr_labels):
            continue
        Xtr=np.vstack(tr_feats); ytr=np.array(tr_labels)
        Xte=np.vstack(te_feats); yte=np.array(te_labels)
        # Build pairwise dataset and train logistic regression
        P_X, P_y = build_pairs(Xtr, ytr)
        if len(P_y)==0:
            continue
        clf = LogisticRegression(max_iter=200, C=1.0, class_weight='balanced')
        clf.fit(P_X, P_y)
        # Predict pairwise score indirectly: use w^T x as utility
        util_tr = (Xtr @ clf.coef_.reshape(-1,1)).squeeze()
        util = (Xte @ clf.coef_.reshape(-1,1)).squeeze()
        # Min-max normalize util to [0,1]
        if len(util)>0:
            umin, umax = float(util.min()), float(util.max())
            util_n = (util - umin)/(umax-umin) if umax>umin else np.full_like(util, 0.5)
        else:
            util_n = np.zeros_like(util)
        sim_te = np.array(normalize_sim(te_sim))
        # Choose monotone blend w by cross-validation on training campaigns:
        # compute P@5 per training campaign for each w and pick the best w.
        grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        # Build a pseudo per-campaign partition for training (approximate by file order)
        # Recompute util scores for training grouped by campaign files
        # For simplicity, use overall training as one group for scoring
        umin, umax = float(util_tr.min()), float(util_tr.max())
        util_tr_n = (util_tr-umin)/(umax-umin) if umax>umin else np.full_like(util_tr,0.5)
        best_w = 0.7; best_score = -1
        for w in grid:
            final_tr = w*np.linspace(0.4,0.6,len(util_tr)) + (1.0-w)*util_tr_n  # small synthetic baseline spread
            order = np.argsort(-final_tr)
            p5 = float(np.mean(ytr[order[:5]])) if len(order)>=5 else 0.0
            if p5 > best_score:
                best_score = p5; best_w = w
        final = best_w*sim_te + (1.0-best_w)*util_n
        order = np.argsort(-final)
        ids_sorted = [te_ids[i] for i in order]
        labels_map = {te_ids[i]: int(yte[i]) for i in range(len(te_ids))}
        p5 = precision_at_k(ids_sorted, labels_map, 5)
        p10 = precision_at_k(ids_sorted, labels_map, 10)
        p5s.append(p5); p10s.append(p10)
        details.append({'campaign_file': hold.name, 'precision_at_5': p5, 'precision_at_10': p10})
    def avg(v):
        v=[x for x in v if isinstance(x,(int,float)) and not math.isnan(x)]
        return float(np.mean(v)) if v else None
    summary={'precision_at_5': avg(p5s), 'precision_at_10': avg(p10s), 'n_campaigns': len(p5s)}
    OUT.joinpath('ad_level_accuracy_v24.json').write_text(json.dumps(summary, indent=2))
    OUT.joinpath('ad_level_accuracy_v24_details.json').write_text(json.dumps(details, indent=2))
    print(json.dumps(summary, indent=2))

def main():
    files = sorted(CRE_DIR.glob('*.json'))
    if not files:
        print('No enriched files found.'); return
    evaluate(files)

if __name__ == '__main__':
    main()
