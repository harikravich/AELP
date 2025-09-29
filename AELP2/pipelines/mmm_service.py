#!/usr/bin/env python3
"""
MMM v1 (Lightweight, dependency-free):

Fits a simple diminishing-returns response curve on daily Google Ads spend and conversions/revenue
using a log-log model with an optional adstock transform. Writes two BigQuery tables:

- <project>.<dataset>.mmm_curves
  Fields: timestamp, channel, window_start, window_end, model, params(JSON),
          spend_grid(JSON), conv_grid(JSON), rev_grid(JSON), diagnostics(JSON)

- <project>.<dataset>.mmm_allocations
  Fields: timestamp, channel, proposed_daily_budget, expected_conversions, expected_revenue,
          expected_cac, constraints(JSON), diagnostics(JSON)

Notes:
- This is a bootstrap MMM that avoids heavy dependencies. It provides usable curves and
  allocations for the Google Ads aggregate (channel='google_ads') and can be extended to
  per-campaign/segment allocations later.
- It reads from the view `<dataset>.ads_campaign_daily` created by create_bq_views.py.
- CAC cap default is 200 (can be overridden by env AELP2_CAC_CAP or CLI).
"""

import os
import json
import math
import argparse
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Tuple

import numpy as np

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def ensure_tables(bq: bigquery.Client, project: str, dataset: str) -> None:
    ds = f"{project}.{dataset}"
    # mmm_curves
    curves_id = f"{ds}.mmm_curves"
    try:
        bq.get_table(curves_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
            bigquery.SchemaField("model", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("params", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("spend_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("conv_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("rev_grid", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("diagnostics", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(curves_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="timestamp"
        )
        bq.create_table(table)

    # mmm_allocations
    allocs_id = f"{ds}.mmm_allocations"
    try:
        bq.get_table(allocs_id)
    except NotFound:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("channel", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("proposed_daily_budget", "FLOAT", mode="REQUIRED"),
            bigquery.SchemaField("expected_conversions", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("expected_revenue", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("expected_cac", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("constraints", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("diagnostics", "JSON", mode="NULLABLE"),
        ]
        table = bigquery.Table(allocs_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY, field="timestamp"
        )
        bq.create_table(table)


def fetch_ads_daily(bq: bigquery.Client, project: str, dataset: str, start: date, end: date) -> List[Dict[str, Any]]:
    """Fetch daily cost and conversions for MMM.

    If env AELP2_MMM_USE_KPI=1 and the KPI-only view `ads_kpi_daily` exists,
    read conversions/revenue from that view so MMM targets the business KPI.
    Otherwise, fall back to `ads_campaign_daily` (all conversions).
    """
    # 0) If an explicit source view is provided, use it. It must have columns:
    #    date, cost, conversions, revenue (revenue may be NULL)
    source_view = os.getenv('AELP2_MMM_SOURCE_VIEW', '').strip()
    if source_view:
        view = f"{project}.{dataset}.{source_view}" if '.' not in source_view else source_view
        try:
            bq.get_table(view)
            sql = f"""
              SELECT DATE(date) AS d,
                     SAFE_CAST(cost AS FLOAT64) AS cost,
                     SAFE_CAST(conversions AS FLOAT64) AS conversions,
                     SAFE_CAST(revenue AS FLOAT64) AS revenue
              FROM `{view}`
              WHERE DATE(date) BETWEEN '{start}' AND '{end}'
              ORDER BY d
            """
            return [dict(r) for r in bq.query(sql).result()]
        except Exception:
            pass

    use_kpi = os.getenv('AELP2_MMM_USE_KPI', '0') == '1'
    if use_kpi:
        # Try KPI-only daily view (created by create_bq_views.py when AELP2_KPI_CONVERSION_ACTION_IDS is set)
        kpi_view = f"{project}.{dataset}.ads_kpi_daily"
        try:
            bq.get_table(kpi_view)
            sql = f"""
              SELECT DATE(date) AS d,
                     SAFE_CAST(cost AS FLOAT64) AS cost,
                     SAFE_CAST(conversions AS FLOAT64) AS conversions,
                     SAFE_CAST(revenue AS FLOAT64) AS revenue
              FROM `{kpi_view}`
              WHERE DATE(date) BETWEEN '{start}' AND '{end}'
              ORDER BY d
            """
            return [dict(r) for r in bq.query(sql).result()]
        except Exception:
            # Fall through to all-conversions view if KPI view not available
            pass
    view = f"{project}.{dataset}.ads_campaign_daily"
    sql = f"""
      SELECT DATE(date) AS d,
             SAFE_CAST(cost AS FLOAT64) AS cost,
             SAFE_CAST(conversions AS FLOAT64) AS conversions,
             SAFE_CAST(revenue AS FLOAT64) AS revenue
      FROM `{view}`
      WHERE DATE(date) BETWEEN '{start}' AND '{end}'
      ORDER BY d
    """
    return [dict(r) for r in bq.query(sql).result()]


def adstock(series: np.ndarray, decay: float) -> np.ndarray:
    if decay <= 0:
        return series.copy()
    out = np.zeros_like(series)
    carry = 0.0
    for i, x in enumerate(series):
        carry = x + decay * carry
        out[i] = carry
    return out


def fit_loglog_response(spend: np.ndarray, y: np.ndarray, X: np.ndarray | None = None) -> Tuple[float, float, float, np.ndarray | None]:
    """Fit log-log model with optional covariates.
    log(y+eps) = a + b*log(spend+eps) + gamma^T X
    Returns (a, b, eps, gamma)."""
    eps = max(1e-6, np.percentile(y[y>0], 5) * 0.01) if np.any(y > 0) else 1e-3
    xs = np.log(spend + eps)
    ys = np.log(y + eps)
    # OLS
    if X is None:
        DM = np.vstack([np.ones_like(xs), xs]).T
    else:
        DM = np.column_stack([np.ones_like(xs), xs, X])
    try:
        coef, _, _, _ = np.linalg.lstsq(DM, ys, rcond=None)
        a, b = coef[0], coef[1]
        gamma = coef[2:] if DM.shape[1] > 2 else None
    except Exception:
        a, b = 0.0, 0.5
        gamma = None
    return a, b, eps, gamma


def predict_loglog(a: float, b: float, eps: float, spend: np.ndarray, X: np.ndarray | None = None) -> np.ndarray:
    extra = 0.0
    if X is not None:
        extra = X @ np.ones(X.shape[1]) * 0.0  # X term handled via gamma in DM during fit; here we pass X_effect externally if needed
    return np.exp(a + b * np.log(spend + eps)) - eps


def select_best_decay(spend: np.ndarray, target: np.ndarray, X: np.ndarray | None = None) -> Tuple[float, Dict[str, float]]:
    """Grid search adstock decay to minimize RMSE in log-log space."""
    best = {
        'decay': 0.0,
        'rmse': float('inf'),
        'a': 0.0,
        'b': 0.0,
        'eps': 1e-6,
    }
    # Allow forcing decay via env (useful for delayed‑reward series where adstock would double‑count)
    opt = os.getenv('AELP2_MMM_FORCE_DECAY')
    grid = [0.0, 0.2, 0.4, 0.6, 0.8] if not opt else [float(opt)]
    for decay in grid:
        s = adstock(spend, decay)
        a, b, eps, gamma = fit_loglog_response(s, target, X)
        # Incorporate covariates at their average level for prediction (ceteris paribus)
        a_eff = a
        if X is not None and gamma is not None and X.size > 0:
            xbar = np.nanmean(X, axis=0)
            try:
                a_eff = float(a + float(np.dot(gamma, xbar)))
            except Exception:
                a_eff = a
        pred = predict_loglog(a_eff, b, eps, s)
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        if rmse < best['rmse'] and b > 0:  # enforce monotonicity
            best.update({'decay': decay, 'rmse': rmse, 'a': a_eff, 'b': b, 'eps': eps})
    return best['decay'], best


def bootstrap_uncertainty(spend: np.ndarray,
                          conv: np.ndarray,
                          decay: float,
                          spend_grid: np.ndarray,
                          n: int | None = None) -> Dict[str, float]:
    """Rudimentary bootstrap to estimate curve uncertainty.
    Returns median relative CI width across grid as 'uncertainty_pct'."""
    if n is None:
        try:
            n = int(os.getenv('AELP2_MMM_BOOTSTRAP_N', '50'))
        except Exception:
            n = 50
    if len(spend) < 10:
        return {'uncertainty_pct': 0.2}
    preds = []
    idx = np.arange(len(spend))
    for _ in range(n):
        samp = np.random.choice(idx, size=len(idx), replace=True)
        s = adstock(spend[samp], decay)
        a, b, eps, _ = fit_loglog_response(s, conv[samp])
        preds.append(predict_loglog(a, b, eps, spend_grid))
    P = np.vstack(preds) if preds else np.zeros((1, len(spend_grid)))
    p10 = np.percentile(P, 10, axis=0)
    p50 = np.percentile(P, 50, axis=0)
    p90 = np.percentile(P, 90, axis=0)
    rel = np.where(p50 > 0, (p90 - p10) / np.maximum(p50, 1e-9), 0.0)
    unc = float(np.median(rel)) if rel.size else 0.2
    # Clamp to reasonable band
    unc = float(max(0.05, min(0.8, unc)))
    return {'uncertainty_pct': unc}


def compute_allocation_from_curve(spend_grid: np.ndarray,
                                  conv_grid: np.ndarray,
                                  cac_cap: float) -> Tuple[float, float, float]:
    """Pick budget on the grid with CAC <= cap and highest conversions.
    Returns (budget, expected_conversions, expected_cac)."""
    best_idx = 0
    best_conv = -1.0
    for i, spend in enumerate(spend_grid):
        conv = conv_grid[i]
        if conv <= 0:
            continue
        cac = spend / max(conv, 1e-9)
        if cac <= cac_cap and conv > best_conv:
            best_conv = conv
            best_idx = i
    budget = float(spend_grid[best_idx])
    conv = float(conv_grid[best_idx])
    cac = budget / max(conv, 1e-9) if conv > 0 else float('inf')
    return budget, conv, cac


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', help='YYYY-MM-DD (default: 90 days ago)')
    p.add_argument('--end', help='YYYY-MM-DD (default: today)')
    p.add_argument('--cac_cap', type=float, default=_get_env_float('AELP2_CAC_CAP', 200.0))
    p.add_argument('--model_label', help='Optional model label to write in curves row')
    p.add_argument('--bootstrap', type=int, help='Override bootstrap replicates (AELP2_MMM_BOOTSTRAP_N)')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')

    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = date.fromisoformat(args.start) if args.start else (end_d - timedelta(days=90))

    bq = bigquery.Client(project=project)
    ensure_tables(bq, project, dataset)

    rows = fetch_ads_daily(bq, project, dataset, start_d, end_d)
    if not rows:
        print('No ads_campaign_daily rows; aborting')
        return

    spend = np.array([float(r['cost'] or 0.0) for r in rows], dtype=float)
    conv = np.array([float(r['conversions'] or 0.0) for r in rows], dtype=float)
    rev = np.array([float(r['revenue'] or 0.0) for r in rows], dtype=float)

    # Optional covariates from mmm_covariates_daily
    cov_view = os.getenv('AELP2_MMM_COVARIATES_VIEW', f'{project}.{dataset}.mmm_covariates_daily')
    X = None
    try:
        if os.getenv('AELP2_MMM_DISABLE_COVARS', '0') == '1':
            raise Exception('covariates disabled by env')
        # Build matrix for dates present in rows
        dates = [str(r['d']) if 'd' in r else str(r['date']) for r in rows]
        q = f"""
          SELECT date,
                 is_avg, lost_is_budget, lost_is_rank, top_is, abs_top_is,
                 hi_begin_checkout, hi_form_submit_enroll, hi_no_purchase_7,
                 affiliate_triggered,
                 promo_share, brand_cost_share,
                 CAST(dow AS FLOAT64) AS dow,
                 CAST(is_weekend AS FLOAT64) AS is_weekend
          FROM `{cov_view}`
          WHERE date BETWEEN DATE('{start_d}') AND DATE('{end_d}')
        """
        cov = {str(r['date']): r for r in bq.query(q).result()}
        feats = []
        for d in dates:
            c = cov.get(d)
            if not c:
                feats.append([None]*12)
            else:
                feats.append([
                    float(c.get('is_avg') or 0.0),
                    float(c.get('lost_is_budget') or 0.0),
                    float(c.get('lost_is_rank') or 0.0),
                    float(c.get('top_is') or 0.0),
                    float(c.get('abs_top_is') or 0.0),
                    float(c.get('hi_begin_checkout') or 0.0),
                    float(c.get('hi_form_submit_enroll') or 0.0),
                    float(c.get('hi_no_purchase_7') or 0.0),
                    float(c.get('affiliate_triggered') or 0.0),
                    float(c.get('promo_share') or 0.0),
                    float(c.get('brand_cost_share') or 0.0),
                    float(c.get('dow') or 0.0),
                    float(c.get('is_weekend') or 0.0),
                ])
        X = np.array(feats, dtype=float)
        # z-score continuous columns (first 10), leave dow/is_weekend as-is
        if X.size > 0:
            Z = X.copy()
            # z-score all continuous features except last two (dow, is_weekend)
            k = Z.shape[1] - 2
            if k > 0:
                mu = Z[:, :k].mean(axis=0)
                sd = Z[:, :k].std(axis=0) + 1e-9
                Z[:, :k] = (Z[:, :k] - mu) / sd
            X = Z
    except Exception:
        X = None

    # Fit with adstock and covariates (if available)
    decay, best = select_best_decay(spend, conv, X)
    s_ast = adstock(spend, decay)
    a, b, eps = best['a'], best['b'], best['eps']

    # Build grids around observed spend range
    smin = max(1.0, np.percentile(spend[spend>0], 10) if np.any(spend>0) else 1.0)
    smax = float(np.percentile(spend, 90) if np.any(spend>0) else 1000.0)
    spend_grid = np.linspace(smin, max(smin, smax), 25)
    # Predict conversions on grid using adstocked spend ~ approximate by assuming steady-state
    # For grid, apply same adstock transform as 1:1 (approx.): s' = spend_grid (steady-state assumption)
    conv_grid = predict_loglog(a, b, eps, spend_grid)
    # Optional uplift priors: scale conversions by segment uplift if enabled
    uplift_used = False
    uplift_covars = []
    if os.getenv('AELP2_MMM_USE_UPLIFT', '0') == '1':
        try:
            usql = f"SELECT segment, score FROM `{project}.{dataset}.segment_scores_daily` WHERE date = CURRENT_DATE() ORDER BY score DESC LIMIT 5"
            cov = [dict(r) for r in bq.query(usql).result()]
            if cov:
                uplift_used = True
                uplift_covars = cov
                factor = 1.0 + min(0.15, float(sum(c.get('score') or 0.0 for c in cov)))
                conv = conv * factor
        except Exception:
            pass
    # Revenue grid: scale by observed avg revenue per conversion (fallback to 0)
    avg_rev = float(np.nan_to_num(rev.sum() / max(conv.sum(), 1e-9))) if conv.sum() > 0 else 0.0
    rev_grid = conv_grid * avg_rev

    # Allocation under CAC cap
    budget, exp_conv, exp_cac = compute_allocation_from_curve(spend_grid, conv_grid, args.cac_cap)
    exp_rev = exp_conv * avg_rev

    # Estimate uncertainty via bootstrap
    if args.bootstrap:
        os.environ['AELP2_MMM_BOOTSTRAP_N'] = str(args.bootstrap)
    unc_diag = bootstrap_uncertainty(spend, conv, decay, spend_grid)

    # Write curves
    curves_tbl = f"{project}.{dataset}.mmm_curves"
    now = datetime.utcnow().isoformat()
    channel_label = os.getenv('AELP2_MMM_CHANNEL_LABEL', 'google_ads')
    model_label = args.model_label or os.getenv('AELP2_MMM_MODEL_LABEL', 'loglog_adstock')
    curves_row = {
        'timestamp': now,
        'channel': channel_label,
        'window_start': str(start_d),
        'window_end': str(end_d),
        'model': model_label,
        'params': json.dumps({'decay': decay, 'a': a, 'b': b, 'eps': eps, 'uplift_used': uplift_used}),
        'spend_grid': json.dumps(list(map(float, spend_grid))),
        'conv_grid': json.dumps(list(map(float, conv_grid))),
        'rev_grid': json.dumps(list(map(float, rev_grid))),
        'diagnostics': json.dumps({'rmse': best['rmse'], 'avg_rev_per_conv': avg_rev, 'uplift_covariates': uplift_covars, **unc_diag}),
    }
    bq.insert_rows_json(curves_tbl, [curves_row])

    # Write allocations
    allocs_tbl = f"{project}.{dataset}.mmm_allocations"
    allocs_row = {
        'timestamp': now,
        'channel': channel_label,
        'proposed_daily_budget': float(budget),
        'expected_conversions': float(exp_conv),
        'expected_revenue': float(exp_rev),
        'expected_cac': float(exp_cac),
        'constraints': json.dumps({'cac_cap': args.cac_cap}),
        'diagnostics': json.dumps({'grid_min': float(spend_grid.min()), 'grid_max': float(spend_grid.max()), **unc_diag}),
    }
    bq.insert_rows_json(allocs_tbl, [allocs_row])

    print(f"MMM curves written to {curves_tbl} and allocations to {allocs_tbl}")


if __name__ == '__main__':
    main()
