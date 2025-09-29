#!/usr/bin/env python3
"""
Check canary readiness gates based on recent fidelity and KPI metrics.

Gates (configurable via env):
  - AELP2_FIDELITY_MAX_MAPE_ROAS (default 0.5)
  - AELP2_FIDELITY_MAX_MAPE_CAC  (default 0.5)
  - AELP2_FIDELITY_MAX_KS_WINRATE (default 0.35)
  - AELP2_MIN_ROAS (default 0.70)
Window: last 14 days by default (AELP2_FIDELITY_WINDOW_DAYS)

Returns non-zero exit if gates not met; prints detailed reasons.
"""

import os
import sys
from datetime import date, timedelta
from google.cloud import bigquery


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def main():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        print("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET", file=sys.stderr)
        sys.exit(2)
    window_days = int(os.getenv("AELP2_FIDELITY_WINDOW_DAYS", "14"))
    min_roas = get_env_float("AELP2_MIN_ROAS", 0.70)
    max_mape_roas = get_env_float("AELP2_FIDELITY_MAX_MAPE_ROAS", 0.5)
    max_mape_cac = get_env_float("AELP2_FIDELITY_MAX_MAPE_CAC", 0.5)
    max_ks = get_env_float("AELP2_FIDELITY_MAX_KS_WINRATE", 0.35)

    bq = bigquery.Client(project=project)
    ds = f"{project}.{dataset}"

    # 1) Fidelity metrics (latest in window)
    sql_fid = f"""
      SELECT timestamp, passed, mape_roas, mape_cac, ks_winrate_vs_impressionshare
      FROM `{ds}.fidelity_evaluations`
      WHERE DATE(timestamp) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {window_days} DAY) AND CURRENT_DATE()
      ORDER BY timestamp DESC
      LIMIT 1
    """
    rows = list(bq.query(sql_fid).result())
    fid_ok = False
    fid_reason = "no fidelity rows"
    if rows:
        r = rows[0]
        mr = float(r.mape_roas) if r.mape_roas is not None else None
        mc = float(r.mape_cac) if r.mape_cac is not None else None
        ks = float(r.ks_winrate_vs_impressionshare) if r.ks_winrate_vs_impressionshare is not None else None
        conds = []
        if mr is not None:
            conds.append(mr <= max_mape_roas)
        if mc is not None:
            conds.append(mc <= max_mape_cac)
        if ks is not None:
            conds.append(ks <= max_ks)
        fid_ok = all(conds) and len(conds) >= 2  # require at least two metrics present
        fid_reason = f"mape_roas={mr}, mape_cac={mc}, ks={ks}, thresholds=({max_mape_roas},{max_mape_cac},{max_ks})"

    # 2) ROAS over last 14 days (KPI daily)
    sql_roas = f"""
      SELECT SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(cost),0)) AS roas
      FROM `{ds}.ads_kpi_daily`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {window_days} DAY) AND CURRENT_DATE()
    """
    roas_rows = list(bq.query(sql_roas).result())
    roas_val = float(roas_rows[0].roas) if roas_rows and roas_rows[0].roas is not None else 0.0
    roas_ok = roas_val >= min_roas

    print(f"[Fidelity] ok={fid_ok} details=({fid_reason})")
    print(f"[ROAS] ok={roas_ok} roas_14d={roas_val:.3f} min={min_roas:.3f}")

    ready = fid_ok and roas_ok
    print(f"[Canary Readiness] ready={ready}")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()

