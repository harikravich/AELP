"""
Drift Monitor utilities for AELP2.

Checks distributional drift between RL win_rate and Ads impression_share and
optionally triggers recalibration or logs safety events.

Env controls (optional):
- AELP2_DRIFT_MONITOR_ENABLE=1
- AELP2_DRIFT_MONITOR_DAYS=14
- AELP2_DRIFT_MONITOR_MAX_KS=0.45
"""

import os
import logging
from typing import Dict, Any

try:
    from google.cloud import bigquery
except Exception as e:
    bigquery = None

logger = logging.getLogger(__name__)


def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def check_winrate_vs_impressionshare_ks(project: str, dataset: str, days: int = 14) -> Dict[str, Any]:
    """Compute KS statistic between RL win_rate and Ads impression_share over last N days.

    Returns a dict with keys: ks_stat, rl_points, ads_points
    """
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery not available")

    bq = bigquery.Client(project=project)
    rl_sql = f"""
        SELECT CAST(AVG(win_rate) AS FLOAT64) AS wr
        FROM `{project}.{dataset}.training_episodes`
        WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND CURRENT_DATE()
        GROUP BY DATE(timestamp)
    """
    ads_sql = f"""
        SELECT CAST(impression_share AS FLOAT64) AS imp_share
        FROM `{project}.{dataset}.ads_campaign_performance`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY) AND CURRENT_DATE()
          AND impression_share IS NOT NULL
    """
    rl_vals = [float(r.wr) for r in bq.query(rl_sql).result()]
    ads_vals = [float(r.imp_share) for r in bq.query(ads_sql).result()]
    try:
        from scipy.stats import ks_2samp
        ks_stat, _ = ks_2samp(rl_vals, ads_vals) if rl_vals and ads_vals else (float('nan'), None)
    except Exception as e:
        logger.warning(f"scipy not available for KS test: {e}")
        ks_stat = float('nan')
    return {"ks_stat": ks_stat, "rl_points": len(rl_vals), "ads_points": len(ads_vals)}


def should_recalibrate(project: str, dataset: str) -> Dict[str, Any]:
    days = int(os.getenv("AELP2_DRIFT_MONITOR_DAYS", "14"))
    max_ks = _get_env_float("AELP2_DRIFT_MONITOR_MAX_KS", 0.45)
    res = check_winrate_vs_impressionshare_ks(project, dataset, days=days)
    res["threshold"] = max_ks
    res["recalibrate"] = (res["ks_stat"] == res["ks_stat"]) and (res["ks_stat"] > max_ks)
    return res
