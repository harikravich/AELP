#!/usr/bin/env python3
"""
Fidelity Evaluation: Compare simulation/RL telemetry vs GA4/Ads (MAPE/RMSE/KS) and write results to BigQuery.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Threshold envs (fail fast if set but not valid):
  - AELP2_FIDELITY_MAX_MAPE_ROAS, AELP2_FIDELITY_MAX_MAPE_CAC (floats)
  - AELP2_FIDELITY_MAX_RMSE_ROAS, AELP2_FIDELITY_MAX_RMSE_CAC (floats)
  - AELP2_FIDELITY_MAX_KS_WINRATE (float) — KS for RL win_rate vs Ads impression_share

Inputs:
- Date range via args: --start YYYY-MM-DD --end YYYY-MM-DD

Behavior:
- Reads RL telemetry from `${BIGQUERY_TRAINING_DATASET}.training_episodes` (aggregated per day)
- Reads Ads aggregates from `${BIGQUERY_TRAINING_DATASET}.ads_campaign_performance`
- Optionally reads GA4 aggregates from `${BIGQUERY_TRAINING_DATASET}.ga4_aggregates` (if present)
- Computes alignment by date; calculates MAPE and RMSE for CAC and ROAS; KS between RL win_rate and Ads impression_share
- Writes a single evaluation row into `${BIGQUERY_TRAINING_DATASET}.fidelity_evaluations`

Notes:
- Fails fast with actionable errors if required tables are missing.
- No dummy data; skips GA4-specific metrics if table is absent (reports accordingly).
"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

try:
    import numpy as np
    from scipy.stats import ks_2samp
except Exception as e:
    raise ImportError(
        f"Fidelity evaluation requires numpy and scipy: {e}. Install with pip."
    )

try:
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
except Exception as e:
    raise ImportError(
        f"google-cloud-bigquery is required: {e}. Install with pip and configure ADC."
    )

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def _required_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _safe_pct_err(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return np.abs((y_pred - y_true) / denom)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = _safe_pct_err(y_true, y_pred)
    return float(np.nanmean(e))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y_pred - y_true) ** 2)))


def fetch_daily_rl_metrics(bq: bigquery.Client, project: str, dataset: str, start: str, end: str) -> List[Dict[str, Any]]:
    table = f"{project}.{dataset}.training_episodes"
    try:
        bq.get_table(table)
    except NotFound:
        raise RuntimeError(
            f"Required table not found: {table}. Ensure orchestrator writes training telemetry to BigQuery."
        )
    sql = f"""
        SELECT
          DATE(timestamp) AS date,
          SUM(spend) AS spend,
          SUM(revenue) AS revenue,
          SUM(conversions) AS conversions,
          SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(spend), 0)) AS roas,
          SAFE_DIVIDE(SUM(spend), NULLIF(SUM(conversions), 0)) AS cac,
          AVG(win_rate) AS avg_win_rate
        FROM `{table}`
        WHERE DATE(timestamp) BETWEEN '{start}' AND '{end}'
        GROUP BY date
        ORDER BY date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if not rows:
        raise RuntimeError(
            f"No RL telemetry rows found in {table} for {start}..{end}. Run training or widen the range."
        )
    return rows


def fetch_daily_ads_metrics(bq: bigquery.Client, project: str, dataset: str, start: str, end: str) -> List[Dict[str, Any]]:
    use_kpi = os.getenv('AELP2_FIDELITY_USE_KPI_ONLY', '0') in ('1', 'true', 'TRUE')
    base = f"{project}.{dataset}"
    table = f"{base}.ads_campaign_performance"
    if use_kpi:
        # Prefer KPI-only view if present
        try:
            bq.get_table(f"{base}.ads_kpi_daily")
            sql = f"""
                SELECT
                  DATE(date) AS date,
                  NULL AS impressions,
                  NULL AS clicks,
                  SUM(cost) AS cost,
                  SUM(conversions) AS conversions,
                  SUM(revenue) AS revenue,
                  NULL AS ctr,
                  NULL AS cvr,
                  SAFE_DIVIDE(SUM(cost), NULLIF(SUM(conversions),0)) AS cac,
                  SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(cost),0)) AS roas,
                  NULL AS impression_share_p50
                FROM `{base}.ads_kpi_daily`
                WHERE DATE(date) BETWEEN '{start}' AND '{end}'
                GROUP BY date
                ORDER BY date
            """
            rows = [dict(r) for r in bq.query(sql).result()]
            if rows:
                return rows
        except Exception:
            pass
    # Fallback to campaign performance table
    try:
        bq.get_table(table)
    except NotFound:
        raise RuntimeError(
            f"Required table not found: {table}. Run Ads ingestion (google_ads_to_bq) first."
        )
    sql = f"""
        SELECT
          DATE(date) AS date,
          SUM(impressions) AS impressions,
          SUM(clicks) AS clicks,
          SUM(cost_micros)/1e6 AS cost,
          SUM(conversions) AS conversions,
          SUM(conversion_value) AS revenue,
          SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
          SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(clicks),0)) AS cvr,
          SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
          SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas,
          APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS impression_share_p50
        FROM `{table}`
        WHERE DATE(date) BETWEEN '{start}' AND '{end}'
        GROUP BY date
        ORDER BY date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if not rows:
        raise RuntimeError(
            f"No Ads rows found in {table} for {start}..{end}. Verify date range and API quotas."
        )
    return rows


def fetch_ads_impression_share_samples(bq: bigquery.Client, project: str, dataset: str, start: str, end: str) -> List[float]:
    table = f"{project}.{dataset}.ads_campaign_performance"
    sql = f"""
        SELECT impression_share
        FROM `{table}`
        WHERE DATE(date) BETWEEN '{start}' AND '{end}'
          AND impression_share IS NOT NULL
    """
    return [float(r[0]) for r in bq.query(sql).result()]


def align_by_date(rl: List[Dict[str, Any]], ads: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rl_by_date = {str(r['date']): r for r in rl}
    ads_by_date = {str(a['date']): a for a in ads}
    common = sorted(set(rl_by_date.keys()) & set(ads_by_date.keys()))
    if not common:
        raise RuntimeError("No overlapping dates between RL and Ads metrics; check date range alignment.")
    rl_roas = np.array([float(rl_by_date[d]['roas'] or 0.0) for d in common])
    ads_roas = np.array([float(ads_by_date[d]['roas'] or 0.0) for d in common])
    rl_cac = np.array([float(rl_by_date[d]['cac']) if rl_by_date[d]['cac'] is not None else np.nan for d in common])
    ads_cac = np.array([float(ads_by_date[d]['cac']) if ads_by_date[d]['cac'] is not None else np.nan for d in common])
    rl_wr = np.array([float(rl_by_date[d]['avg_win_rate'] or 0.0) for d in common])
    ads_is = np.array([float(ads_by_date[d]['impression_share_p50'] or 0.0) for d in common])
    return rl_roas, ads_roas, rl_cac, ads_cac, (rl_wr - rl_wr + ads_is)  # ads_is pass-through for date alignment


def write_result(bq: bigquery.Client, project: str, dataset: str, row: Dict[str, Any]) -> None:
    table_id = f"{project}.{dataset}.fidelity_evaluations"
    # Ensure table exists
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("start_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("end_date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("mape_roas", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rmse_roas", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("mape_cac", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("rmse_cac", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("ks_winrate_vs_impressionshare", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("passed", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("details", "JSON", mode="NULLABLE"),
    ]
    try:
        bq.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="timestamp"
        )
        bq.create_table(table)
        logger.info(f"Created table: {table_id}")
    errors = bq.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    # Validate dates
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")

    project = _required_env("GOOGLE_CLOUD_PROJECT")
    dataset = _required_env("BIGQUERY_TRAINING_DATASET")

    # Thresholds (optional; if set, we gate; otherwise we only report)
    def _getf(name: str):
        val = os.getenv(name)
        return float(val) if val is not None else None
    max_mape_roas = _getf('AELP2_FIDELITY_MAX_MAPE_ROAS')
    max_rmse_roas = _getf('AELP2_FIDELITY_MAX_RMSE_ROAS')
    max_mape_cac = _getf('AELP2_FIDELITY_MAX_MAPE_CAC')
    max_rmse_cac = _getf('AELP2_FIDELITY_MAX_RMSE_CAC')
    max_ks_wr = _getf('AELP2_FIDELITY_MAX_KS_WINRATE')

    bq = bigquery.Client(project=project)

    rl_daily = fetch_daily_rl_metrics(bq, project, dataset, args.start, args.end)
    ads_daily = fetch_daily_ads_metrics(bq, project, dataset, args.start, args.end)
    rl_roas, ads_roas, rl_cac, ads_cac, ads_is = align_by_date(rl_daily, ads_daily)

    # Compute metrics
    mape_roas_val = _mape(ads_roas, rl_roas)
    rmse_roas_val = _rmse(ads_roas, rl_roas)

    # CAC arrays may contain NaN (no conversions); filter pairwise
    mask = ~np.isnan(ads_cac) & ~np.isnan(rl_cac)
    mape_cac_val = _mape(ads_cac[mask], rl_cac[mask]) if mask.any() else np.nan
    rmse_cac_val = _rmse(ads_cac[mask], rl_cac[mask]) if mask.any() else np.nan

    # KS between RL win_rate and Ads impression_share distributions
    # Use all RL daily win_rate points, and raw Ads impression_share samples over the range
    rl_wr_series = np.array([float(r['avg_win_rate'] or 0.0) for r in rl_daily], dtype=float)
    ads_is_samples = np.array(fetch_ads_impression_share_samples(bq, project, dataset, args.start, args.end), dtype=float)
    if rl_wr_series.size == 0 or ads_is_samples.size == 0:
        ks_stat_val = np.nan
    else:
        ks_stat_val, _ = ks_2samp(rl_wr_series, ads_is_samples)

    # Sanitize: convert NaN to None for BQ JSON
    def _san(x):
        try:
            if x != x:  # NaN
                return None
            return float(x)
        except Exception:
            return None

    mape_roas = _san(mape_roas_val)
    rmse_roas = _san(rmse_roas_val)
    mape_cac = _san(mape_cac_val)
    rmse_cac = _san(rmse_cac_val)
    ks_stat = _san(ks_stat_val)

    # Gate evaluation
    passed = True
    details: Dict[str, Any] = {
        'mape_roas': mape_roas,
        'rmse_roas': rmse_roas,
        'mape_cac': mape_cac,
        'rmse_cac': rmse_cac,
        'ks_winrate_vs_impressionshare': ks_stat,
        'eval_dates': {'start': args.start, 'end': args.end},
    }

    # Optional: include GA4 aggregates if present for context
    try:
        ga4_sql = f"""
            SELECT DATE(date) AS date, SUM(sessions) AS sessions, SUM(conversions) AS conversions
            FROM `{project}.{dataset}.ga4_daily`
            WHERE DATE(date) BETWEEN '{args.start}' AND '{args.end}'
            GROUP BY date
            ORDER BY date
        """
        ga4_rows = [dict(r) for r in bq.query(ga4_sql).result()]
        details['ga4_context'] = {'present': True, 'days': len(ga4_rows)}
    except Exception:
        details['ga4_context'] = {'present': False}

    # Apply thresholds if provided
    def check(name: str, val: float, thr: float):
        nonlocal passed
        if thr is not None and not np.isnan(val) and val > thr:
            passed = False
            details.setdefault('threshold_violations', []).append({name: {'value': val, 'max': thr}})

    if mape_roas is not None:
        check('mape_roas', mape_roas, max_mape_roas)
    if rmse_roas is not None:
        check('rmse_roas', rmse_roas, max_rmse_roas)
    if mape_cac is not None:
        check('mape_cac', mape_cac, max_mape_cac)
    if rmse_cac is not None:
        check('rmse_cac', rmse_cac, max_rmse_cac)
    if ks_stat is not None:
        check('ks_winrate_vs_impressionshare', ks_stat, max_ks_wr)

    # Write result
    # Ensure details is JSON-encoded string for BigQuery JSON field
    details_str = json.dumps(details)

    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'start_date': args.start,
        'end_date': args.end,
        'mape_roas': mape_roas,
        'rmse_roas': rmse_roas,
        'mape_cac': mape_cac,
        'rmse_cac': rmse_cac,
        'ks_winrate_vs_impressionshare': ks_stat,
        'passed': bool(passed),
        'details': details_str,
    }
    write_result(bq, project, dataset, row)
    status = "PASSED" if passed else "FAILED"
    def fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "null"
    logger.info(
        "Fidelity evaluation %s: ROAS(MAPE=%s, RMSE=%s), CAC(MAPE=%s, RMSE=%s), KS=%s"
        % (status, fmt(mape_roas), fmt(rmse_roas), fmt(mape_cac), fmt(rmse_cac), fmt(ks_stat))
    )


if __name__ == "__main__":
    main()
