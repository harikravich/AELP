#!/usr/bin/env python3
from __future__ import annotations
"""
Backtest Auction-aware ranking accuracy on historical campaigns using BigQuery metrics.

For each file in AELP2/reports/creative/*.json:
 - Read campaign_id and creative_id list with actual_label.
 - Query BigQuery ads_ad_performance to aggregate impressions, clicks, conversions, cost per ad_id for that campaign over the last 60 days.
 - Derive CTR, CVR; run AuctionGym to estimate CPC and CAC; rank creatives by lowest CAC.
 - Compute Precision@10 vs actual_label in the JSON file.

Outputs: AELP2/reports/auction_backtest_summary.json
"""
import os, json, math
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'AELP2' / 'reports'
CREATIVE_DIR = REPORTS / 'creative'
OUT = REPORTS / 'auction_backtest_summary.json'

import sys
sys.path.insert(0, str(ROOT))
from auction_gym_integration import AuctionGymWrapper
from google.cloud import bigquery
import numpy as np
try:
    import joblib  # for loading CPC model
except Exception:
    joblib = None


def precision_at_k(sorted_ids: List[str], labels: Dict[str, int], k: int=10) -> float:
    topk = sorted_ids[:k]
    if not topk: return float('nan')
    return sum(labels.get(cid,0) for cid in topk) / len(topk)


def fetch_metrics_for_campaign(bq: bigquery.Client, project: str, dataset: str, campaign_id: str) -> Dict[str, Dict[str, float]]:
    # Use Meta table (string ids; fields: impressions, clicks, cost, conversions)
    sql = f"""
      SELECT ad_id AS ad_id,
             SUM(CAST(impressions AS INT64)) AS imps,
             SUM(CAST(clicks AS INT64)) AS clicks,
             SUM(CAST(conversions AS FLOAT64)) AS conv,
             SUM(CAST(cost AS FLOAT64)) AS cost
      FROM `{project}.{dataset}.meta_ad_performance`
      WHERE campaign_id = '{campaign_id}'
        AND DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND CURRENT_DATE()
      GROUP BY ad_id
    """
    rows = list(bq.query(sql).result())
    out = {}
    for r in rows:
        imps = float(r['imps'] or 0.0)
        clicks = float(r['clicks'] or 0.0)
        conv = float(r['conv'] or 0.0)
        cost = float(r['cost'] or 0.0)
        out[str(r['ad_id'])] = {
            'imps': imps,
            'clicks': clicks,
            'conv': conv,
            'cost': cost,
        }
    return out


def fetch_metrics_by_place(bq: bigquery.Client, project: str, dataset: str, campaign_id: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Optional: per-ad placement/device breakdown if the by_place table exists."""
    sql = f"""
      SELECT ad_id AS ad_id,
             COALESCE(publisher_platform,'unknown') AS pp,
             COALESCE(placement,'unknown') AS place,
             COALESCE(device,'unknown') AS dev,
             SUM(CAST(impressions AS INT64)) AS imps,
             SUM(CAST(clicks AS INT64)) AS clicks,
             SUM(CAST(conversions AS FLOAT64)) AS conv,
             SUM(CAST(cost AS FLOAT64)) AS cost
      FROM `{project}.{dataset}.meta_ad_performance_by_place`
      WHERE campaign_id = '{campaign_id}'
        AND DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY) AND CURRENT_DATE()
      GROUP BY ad_id, pp, place, dev
    """
    try:
        rows = list(bq.query(sql).result())
    except Exception:
        return {}
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in rows:
        key = f"{r['pp']}|{r['place']}|{r['dev']}"
        a = str(r['ad_id'])
        m = out.setdefault(a, {})
        m[key] = {
            'imps': float(r['imps'] or 0.0),
            'clicks': float(r['clicks'] or 0.0),
            'conv': float(r['conv'] or 0.0),
            'cost': float(r['cost'] or 0.0),
        }
    return out


def derive_rates(m: Dict[str, float]) -> Tuple[float,float]:
    imps = m.get('imps',0.0); clicks = m.get('clicks',0.0); conv = m.get('conv',0.0)
    # Empirical
    ctr_emp = clicks / imps if imps>0 else 0.0
    cvr_emp = conv / clicks if clicks>0 else 0.0
    # Bayesian shrinkage (light priors)
    # CTR prior: 0.5% with pseudo-counts ~ 50 impressions
    ctr_prior_a, ctr_prior_b = 0.25, 49.75
    ctr = (clicks + ctr_prior_a) / (imps + ctr_prior_a + ctr_prior_b) if imps>0 else (ctr_prior_a/(ctr_prior_a+ctr_prior_b))
    # CVR prior: 2% with pseudo-counts ~ 50 clicks
    cvr_prior_a, cvr_prior_b = 1.0, 49.0
    cvr = (conv + cvr_prior_a) / (clicks + cvr_prior_a + cvr_prior_b) if clicks>0 else (cvr_prior_a/(cvr_prior_a+cvr_prior_b))
    # Clamp to reasonable ranges
    ctr = max(1e-5, min(0.2, ctr))
    cvr = max(0.001, min(0.2, cvr))
    return ctr, cvr


def predict_cac_auction(auction: AuctionGymWrapper, ctr: float, cvr: float, rounds: int=1000) -> tuple[float,float]:
    # Quality proxy from CTR; bid as fraction of query_value (consistent with sim scripts)
    quality = 0.5 + 2.0 * min(0.5, ctr / 0.05)
    query_value = quality * 10.0
    our_bid = min(5.8, max(0.2, 0.4 * query_value))
    imps = clicks = 0
    spend = 0.0
    import random as _rnd
    for _ in range(rounds):
        res = auction.run_auction(our_bid, quality, context={'estimated_ctr': ctr})
        if res.won:
            imps += 1
            if _rnd.random() < ctr:
                clicks += 1
                spend += res.price_paid
    if clicks == 0:
        return float('inf'), float('inf')
    cpc = spend / clicks
    cac = cpc / max(cvr, 1e-6)
    return cpc, cac


def load_cpc_model():
    """Load CPC regression model and feature schema if present."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    p = root / 'AELP2' / 'reports' / 'models' / 'cpc_model.joblib'
    if joblib and p.exists():
        try:
            obj = joblib.load(p)
            return obj.get('model'), obj.get('features', [])
        except Exception:
            return None, []
    return None, []


def extract_features_for_ad(ad_id: str, feature_keys: list[str]) -> np.ndarray:
    """Mirror the trainer's feature extraction using creative_objects JSON."""
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]
    p = root / 'AELP2' / 'reports' / 'creative_objects' / f"{ad_id}.json"
    feats = { 'bias': 1.0 }
    if p.exists():
        try:
            d = json.loads(p.read_text())
        except Exception:
            d = {}
        cr = (d.get('creative') or {})
        afs = (cr.get('asset_feed_spec') or {})
        videos = afs.get('videos') or []
        if videos: feats['is_video'] = 1.0
        feats['n_titles'] = float(len(afs.get('titles') or []))
        feats['n_bodies'] = float(len(afs.get('bodies') or []))
        for cta in (afs.get('call_to_action_types') or []):
            feats[f"cta_{str(cta).lower()}"] = 1.0
        rules = afs.get('asset_customization_rules') or []
        for r in rules:
            cs = (r.get('customization_spec') or {})
            for pos in (cs.get('facebook_positions') or []):
                feats[f"fbpos_{pos}"] = 1.0
            for pos in (cs.get('instagram_positions') or []):
                feats[f"igpos_{pos}"] = 1.0
            for pos in (cs.get('audience_network_positions') or []):
                feats[f"anpos_{pos}"] = 1.0
            for plat in (cs.get('publisher_platforms') or []):
                feats[f"plat_{plat}"] = 1.0
        opt = cr.get('asset_feed_spec', {}).get('optimization_type') or cr.get('optimization_type')
        if opt:
            feats[f"opt_{str(opt).lower()}"] = 1.0
    # Vectorize into feature_keys order
    X = np.zeros((1, len(feature_keys)), dtype=np.float32)
    for i, k in enumerate(feature_keys):
        X[0, i] = float(feats.get(k, 0.0))
    return X


def load_placement_calibrators():
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]
    p = root / 'AELP2' / 'reports' / 'placement_calibrators.json'
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return float(d.get('global_cvr') or 0.01), (d.get('calibrators') or {})
        except Exception:
            return None, {}
    return None, {}


def load_baseline_cpc_map():
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]
    p = root / 'AELP2' / 'reports' / 'us_meta_baselines_by_place.json'
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text())
    except Exception:
        return {}
    items = d.get('items') or {}
    cpc_map = {}
    for key, vals in items.items():
        try:
            cpm = float(vals.get('cpm_p50') or 0.0)
            ctr = float(vals.get('ctr_p50') or 0.0)
            if cpm>0 and ctr>0:
                cpc = (cpm/1000.0) / ctr
                cpc_map[key] = float(cpc)
        except Exception:
            continue
    return cpc_map


def guess_placement_key(ad_id: str) -> str:
    """Approximate a placement key from creative_objects asset_customization_rules."""
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[2]
    p = root / 'AELP2' / 'reports' / 'creative_objects' / f"{ad_id}.json"
    if not p.exists():
        return 'unknown|unknown|unknown'
    try:
        d = json.loads(p.read_text())
    except Exception:
        return 'unknown|unknown|unknown'
    rules = ((d.get('creative') or {}).get('asset_feed_spec') or {}).get('asset_customization_rules') or []
    if not rules:
        return 'unknown|unknown|unknown'
    cs = (rules[0].get('customization_spec') or {})
    plat = (cs.get('publisher_platforms') or ['unknown'])[0]
    # prefer specific position lists in order: facebook, instagram, audience_network
    pos = None
    for key in ('facebook_positions','instagram_positions','audience_network_positions'):
        arr = cs.get(key) or []
        if arr:
            pos = arr[0]; break
    if not pos:
        pos = 'unknown'
    # no device info => unknown
    return f"{plat}|{pos}|unknown"


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT','aura-thrive-platform')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET','gaelp_training')
    bq = bigquery.Client(project=project)
    auction = AuctionGymWrapper({'competitors': {'count': 6}, 'num_slots': 4})
    cpc_model, cpc_keys = load_cpc_model()
    global_cvr, placemap = load_placement_calibrators()
    base_cpc_map = load_baseline_cpc_map()

    precisions = []
    details = []
    for f in sorted(CREATIVE_DIR.glob('*.json')):
        d = json.loads(f.read_text())
        cid = str(d.get('campaign_id') or f.stem)
        items = d.get('items') or []
        if not items:
            continue
        labels = {str(it['creative_id']): int(it.get('actual_label') or 0) for it in items}
        metrics = fetch_metrics_for_campaign(bq, project, dataset, cid)
        scores = []
        pred_cpcs = []
        for ad_id in labels.keys():
            m = metrics.get(str(ad_id))
            if not m:
                continue
            # ignore very low evidence to reduce noise
            if m.get('imps',0.0) < 500 or m.get('clicks',0.0) < 20:
                continue
            ctr, cvr = derive_rates(m)
            # placement CVR calibration (if available)
            try:
                key = guess_placement_key(str(ad_id))
                mult = (placemap.get(key) or {}).get('cvr')
                if isinstance(mult, (int,float)) and mult and math.isfinite(mult):
                    cvr = max(1e-5, min(0.5, cvr * float(mult)))
            except Exception:
                pass
            cpc = None; cac = None
            if cpc_model is not None and cpc_keys:
                try:
                    X = extract_features_for_ad(str(ad_id), cpc_keys)
                    cpc_hat = float(cpc_model.predict(X)[0])
                    if math.isfinite(cpc_hat) and cpc_hat>0:
                        # Market pressure floor using baseline CPC for guessed placement
                        key = guess_placement_key(str(ad_id))
                        platpos = "|".join(key.split('|')[:2])  # platform|position
                        comp_cpc = base_cpc_map.get(platpos.replace('|','/'))
                        if isinstance(comp_cpc, (int,float)) and comp_cpc>0:
                            cpc = max(cpc_hat, 0.7*float(comp_cpc))
                        else:
                            cpc = cpc_hat
                        cac = cpc_hat / max(cvr, 1e-6)
                except Exception:
                    pass
            if cpc is None:
                cpc, cac = predict_cac_auction(auction, ctr, cvr, rounds=800)
            if math.isfinite(cpc):
                pred_cpcs.append(cpc)
            scores.append((ad_id, cac))
        # Per-campaign CPC calibration to align sim CPC with actual CPC
        ranked = []
        if scores:
            cal_scores = scores
            if pred_cpcs:
                mean_pred_cpc = sum(pred_cpcs)/len(pred_cpcs)
                clicks_sum = sum(m.get('clicks',0.0) for m in metrics.values())
                cost_sum = sum(m.get('cost',0.0) for m in metrics.values())
                camp_cpc = (cost_sum / clicks_sum) if clicks_sum>0 else None
                if camp_cpc and mean_pred_cpc>0:
                    s = camp_cpc / mean_pred_cpc
                    cal_scores = [(ad, (cac*s if math.isfinite(cac) else cac)) for ad, cac in scores]
            ranked = [ad for ad,_ in sorted(cal_scores, key=lambda t: t[1])]
        p10 = precision_at_k(ranked, labels, 10)
        if isinstance(p10,(int,float)) and not math.isnan(p10):
            precisions.append(p10)
        details.append({'campaign_id': cid, 'n': len(items), 'p10': p10})

    summary = {
        'precision_at_10': round(sum(precisions)/len(precisions), 3) if precisions else None,
        'n_campaigns': len(precisions),
        'details': details[:50]
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
