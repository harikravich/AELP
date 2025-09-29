#!/usr/bin/env python3
from __future__ import annotations
import os, json, math
from pathlib import Path
from typing import List, Dict
import numpy as np

# Light ML stack
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CRE_DIR = ROOT / 'AELP2' / 'reports' / 'creative_with_dna'
if not CRE_DIR.exists():
    CRE_DIR = ROOT / 'AELP2' / 'reports' / 'creative'
OUT = ROOT / 'AELP2' / 'reports'
OUT.mkdir(parents=True, exist_ok=True)


def load_campaign_files() -> List[Path]:
    return sorted(CRE_DIR.glob('*.json'))


def top_placement_keys(files: List[Path], k: int = 8) -> List[str]:
    counts = {}
    for f in files:
        data = json.loads(f.read_text())
        for it in (data.get('items') or []):
            for key in (it.get('placement_mix') or {}).keys():
                counts[key] = counts.get(key, 0) + 1
    keys = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
    return keys[:k]


def featurize_item(it: dict, keys: List[str]) -> List[float]:
    x: List[float] = []
    mix = it.get('placement_mix') or {}
    for k in keys:
        x.append(float(mix.get(k, 0.0)))
    dna = (it.get('dna') or {})
    fmt = dna.get('format') or ''
    tags = set(dna.get('tags') or [])
    # one-hot simple features
    x += [1.0 if fmt == 'vertical-video' else 0.0,
          1.0 if fmt == 'video' else 0.0]
    x += [1.0 if 'balance_parental' in tags else 0.0,
          1.0 if 'breach_alerts' in tags else 0.0,
          1.0 if 'speed_benefit' in tags else 0.0]
    return x

def reconstruct_clicks_and_prior(items: List[dict]) -> float:
    # Estimate campaign prior CVR from sim_score = clicks * prior_cvr
    ratios = []
    for it in items:
        s = float(it.get('sim_score',0.0)); tc = float(it.get('test_clicks') or 0.0)
        if tc > 0: ratios.append(s/max(tc,1e-6))
    if not ratios:
        # fallback from all items using median ratio of sim_score to max(sim_score) as proxy
        return 0.01
    ratios = [r for r in ratios if math.isfinite(r) and r>0]
    return float(np.median(ratios)) if ratios else 0.01


def calibrate_and_eval(files: List[Path]):
    plac_keys = top_placement_keys(files)
    all_metrics = []
    ys_all, ps_all = [], []
    # Leave-one-campaign-out CV
    for hold in files:
        train_X, train_y, train_s = [], [], []
        test_X, test_y, test_s = [], [], []
        test_ids = []
        for f in files:
            data = json.loads(f.read_text())
            items = data.get('items') or []
            # reconstruct campaign prior cvr
            prior_cvr = reconstruct_clicks_and_prior(items)
            for it in items:
                y = int(it.get('actual_label', 0))
                s = float(it.get('sim_score', 0.0))
                # hierarchical adjustment: if train stats exist, update posterior CVR
                trc = float(it.get('train_clicks') or 0.0)
                trp = float(it.get('train_purch') or 0.0)
                # derive test clicks if provided or from sim_score/prior
                tclk = float(it.get('test_clicks') or (s/max(prior_cvr,1e-6)))
                a0 = prior_cvr * 100.0
                b0 = (1-prior_cvr) * 100.0
                e_cvr = (a0 + trp) / max(1e-6, (a0 + b0 + trc))
                hier_score = tclk * e_cvr
                x = featurize_item(it, plac_keys)
                if f == hold:
                    test_X.append(x + [hier_score]); test_y.append(y); test_s.append(s); test_ids.append(it.get('creative_id'))
                else:
                    train_X.append(x + [hier_score]); train_y.append(y); train_s.append(s)
        if not test_X or sum(train_y) == 0:
            continue
        Xtr = np.array(train_X); ytr = np.array(train_y); Str = np.array(train_s)
        # Train a simple L2 logistic model using [features, sim_score]
        # Features + original sim_score
        Xtr_full = np.hstack([Xtr, Str.reshape(-1,1)])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        clf.fit(Xtr_full, ytr)
        # Raw model probabilities on train for isotonic calibration
        p_tr = clf.predict_proba(Xtr_full)[:,1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(p_tr, ytr)

        # Evaluate on held-out
        Xte = np.array(test_X); Ste = np.array(test_s)
        Xte_full = np.hstack([Xte, Ste.reshape(-1,1)])
        p_raw = clf.predict_proba(Xte_full)[:,1]
        p_cal = iso.transform(p_raw)

        # Metrics
        order = np.argsort(-p_cal)
        top5 = order[:5]; top10 = order[:10]
        p_at_5 = float(np.mean([test_y[i] for i in top5])) if len(order)>=5 else float('nan')
        p_at_10 = float(np.mean([test_y[i] for i in top10])) if len(order)>=10 else float('nan')
        # pairwise win-rate
        wins = tot = 0
        for i in range(len(order)):
            for j in range(i+1, len(order)):
                yi, yj = test_y[order[i]], test_y[order[j]]
                if yi == yj: continue
                tot += 1
                if yi > yj: wins += 1
        pw = wins/tot if tot else float('nan')

        all_metrics.append({'campaign_file': hold.name, 'precision_at_5': p_at_5, 'precision_at_10': p_at_10, 'pairwise_win_rate': pw})
        ys_all.extend(test_y)
        ps_all.extend(p_cal)

    # Aggregate
    def avg(vals):
        vals = [v for v in vals if v==v]  # drop NaN
        return float(np.mean(vals)) if vals else None
    p5 = avg([m['precision_at_5'] for m in all_metrics])
    p10 = avg([m['precision_at_10'] for m in all_metrics])
    pw = avg([m['pairwise_win_rate'] for m in all_metrics])
    # Stability under small noise in sim_score feature
    rng = np.random.default_rng(0)
    stability_scores = []
    for f in files:
        data = json.loads(f.read_text()); items = data.get('items') or []
        if len(items) < 10: continue
        plac_keys = top_placement_keys([f])
        X = np.array([featurize_item(it, plac_keys) for it in items]); S = np.array([it.get('sim_score',0.0) for it in items]).reshape(-1,1)
        X_full = np.hstack([X, S])
        # simple ranking by S; add noise and measure overlap@10
        base_order = np.argsort(-S.squeeze())[:10]
        overlaps = []
        for _ in range(20):
            noise = rng.normal(0, 0.05*np.maximum(1e-6,S), size=S.shape)
            order = np.argsort(-(S+noise).squeeze())[:10]
            overlaps.append(len(set(base_order).intersection(set(order)))/10.0)
        stability_scores.append(float(np.mean(overlaps)))
    stability = avg(stability_scores)
    summary = {'precision_at_5': p5, 'precision_at_10': p10, 'pairwise_win_rate': pw, 'stability_top10': stability, 'n_campaigns': len(all_metrics), 'n_samples': len(ys_all)}

    # Reliability plot
    if ys_all and ps_all:
        bins = np.linspace(0,1,11)
        inds = np.digitize(ps_all, bins) - 1
        prob, true = [], []
        for b in range(10):
            sel = [i for i,z in enumerate(inds) if z==b]
            if not sel: continue
            prob.append((bins[b]+bins[b+1])/2)
            true.append(float(np.mean([ys_all[i] for i in sel])))
        plt.figure(figsize=(4,4))
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.plot(prob, true, 'o-', label='isotonic-calibrated')
        plt.xlabel('Predicted probability'); plt.ylabel('Empirical rate'); plt.title('Ad-level Reliability (v2.2)')
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT / 'ad_calibration_reliability.png', dpi=120)
        plt.close()

    OUT.joinpath('ad_level_accuracy_v22.json').write_text(json.dumps(summary, indent=2))
    OUT.joinpath('ad_level_accuracy_v22_details.json').write_text(json.dumps(all_metrics, indent=2))
    # Markdown report
    md = []
    md.append('# Ad-Level Calibration v2.2 (Offline)')
    md.append('')
    md.append(f"- Precision@5: {summary['precision_at_5']}")
    md.append(f"- Precision@10: {summary['precision_at_10']}")
    md.append(f"- Pairwise win-rate: {summary['pairwise_win_rate']}")
    md.append(f"- Top-10 stability (noise 5% of score): {summary.get('stability_top10')}")
    md.append(f"- Campaigns: {summary['n_campaigns']}  Samples: {summary['n_samples']}")
    md.append('')
    md.append('![Reliability](ad_calibration_reliability.png)')
    OUT.joinpath('ad_calibration_v22.md').write_text('\n'.join(md))
    print(json.dumps(summary, indent=2))


def main():
    files = load_campaign_files()
    if not files:
        print('No campaign files found for ad-level calibration.')
        return
    calibrate_and_eval(files)


if __name__ == '__main__':
    main()
