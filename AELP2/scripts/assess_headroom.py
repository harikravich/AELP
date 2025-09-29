#!/usr/bin/env python3
"""
Assess optimization headroom from real Ads + KPI data in BigQuery.

Outputs:
- Baseline KPI CAC/ROAS over last 30d
- % spend with zero KPI conversions
- % spend with CAC > 1.5× median KPI CAC (tail)
- Impression share hints: overbidding vs budget-constrained spend shares
- Top campaigns in high-CAC and zero-conversion buckets

Env:
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET (required)
  AELP2_KPI_CONVERSION_ACTION_IDS (optional comma list; auto-select top 3 by revenue if absent)

Usage:
  GOOGLE_CLOUD_PROJECT=... BIGQUERY_TRAINING_DATASET=... python3 -m AELP2.scripts.assess_headroom
"""

import os
from typing import List, Tuple
import shutil
import subprocess
import json
from google.cloud import bigquery


def get_kpi_action_ids(bq: bigquery.Client, project: str, dataset: str) -> List[str]:
    env_ids = os.getenv("AELP2_KPI_CONVERSION_ACTION_IDS")
    if env_ids:
        return [x.strip() for x in env_ids.split(',') if x.strip()]
    sql = f"""
        SELECT CAST(conversion_action_id AS STRING) AS id
        FROM `{project}.{dataset}.ads_conversion_action_stats`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY) AND CURRENT_DATE()
        GROUP BY id
        ORDER BY SUM(conversion_value) DESC
        LIMIT 3
    """
    return [r.id for r in bq.query(sql).result()]


def _query_rows_with_fallback(project: str, sql: str):
    """Run a query and return list of dict rows.

    Tries python BigQuery client first; if permission error, falls back to
    `bq query --format=json` using the caller's gcloud credentials.
    """
    try:
        client = bigquery.Client(project=project)
        rows = list(client.query(sql).result())
        # Convert Row to dict
        return [dict(r) for r in rows]
    except Exception as e:
        msg = str(e)
        if not any(s in msg for s in ("403", "Access Denied", "Forbidden")) or shutil.which("bq") is None:
            raise
        cmd = [
            "bq",
            f"--project_id={project}",
            "query",
            "--use_legacy_sql=false",
            "--quiet",
            "--format=json",
            sql,
        ]
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        # bq --format=json returns a list of rows as dicts
        if isinstance(data, list):
            return data
        # Fallback if API-style dict is returned
        if isinstance(data, dict) and "rows" in data:
            result = []
            schema_fields = [f["name"] for f in data.get("schema", {}).get("fields", [])]
            for row in data.get("rows", []):
                values = [cell.get("v") for cell in row.get("f", [])]
                result.append({k: v for k, v in zip(schema_fields, values)})
            return result
        raise RuntimeError("Unexpected bq JSON output format")


def assess(project: str, dataset: str):
    # BigQuery client is only used for environment inspection; queries use fallback helper
    bq = bigquery.Client(project=project)
    kpi_ids = get_kpi_action_ids(bq, project, dataset)
    kpi_ids_csv = ",".join([f"'{_id}'" for _id in kpi_ids]) if kpi_ids else "''"

    print("KPI conversion_action_ids:", kpi_ids or "<none>")

    sql = f"""
    -- KPI by-day by-campaign
    WITH kpi AS (
      SELECT DATE(a.date) AS date, a.campaign_id,
             SUM(s.conversions) AS kpi_conversions,
             SUM(s.conversion_value) AS kpi_revenue
      FROM `{project}.{dataset}.ads_conversion_action_stats` s
      JOIN `{project}.{dataset}.ads_campaign_performance` a
        ON DATE(a.date) = DATE(s.date) AND a.campaign_id = s.campaign_id
      WHERE DATE(a.date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
        AND s.conversion_action_id IN ({kpi_ids_csv})
      GROUP BY date, campaign_id
    ),
    cost AS (
      SELECT DATE(date) AS date, campaign_id, SUM(cost_micros)/1e6 AS cost
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      GROUP BY date, campaign_id
    ),
    daily AS (
      SELECT c.date, c.campaign_id, c.cost,
             COALESCE(k.kpi_conversions, 0) AS conv,
             COALESCE(k.kpi_revenue, 0) AS revenue
      FROM cost c
      LEFT JOIN kpi k USING (date, campaign_id)
    ),
    agg AS (
      SELECT campaign_id,
             SUM(cost) AS cost,
             SUM(conv) AS conv,
             SUM(revenue) AS revenue,
             SAFE_DIVIDE(SUM(cost), NULLIF(SUM(conv),0)) AS cac,
             SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(cost),0)) AS roas
      FROM daily
      GROUP BY campaign_id
    ),
    totals AS (
      SELECT SUM(cost) AS total_cost,
             SUM(conv) AS total_conv,
             SUM(revenue) AS total_revenue,
             SAFE_DIVIDE(SUM(cost), NULLIF(SUM(conv),0)) AS baseline_cac,
             SAFE_DIVIDE(SUM(revenue), NULLIF(SUM(cost),0)) AS baseline_roas
      FROM agg
    ),
    stats AS (
      SELECT APPROX_QUANTILES(cac, 100)[OFFSET(50)] AS cac_median
      FROM agg WHERE conv > 0 AND cac IS NOT NULL
    ),
    flags AS (
      SELECT a.*, s.cac_median,
             CASE WHEN a.conv = 0 THEN 1 ELSE 0 END AS is_zero_conv,
             CASE WHEN a.conv > 0 AND a.cac > 1.5*s.cac_median THEN 1 ELSE 0 END AS is_high_cac
      FROM agg a CROSS JOIN stats s
    ),
    shares AS (
      SELECT
        SAFE_DIVIDE(SUM(CASE WHEN is_zero_conv=1 THEN cost END), NULLIF(SUM(cost),0)) AS pct_spend_zero_conv,
        SAFE_DIVIDE(SUM(CASE WHEN is_high_cac=1 THEN cost END), NULLIF(SUM(cost),0)) AS pct_spend_high_cac
      FROM flags
    ),
    is_hints AS (
      SELECT campaign_id,
             SUM(cost_micros)/1e6 AS cost,
             AVG(impression_share) AS is_avg,
             SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS roas
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      GROUP BY campaign_id
    ),
    is_flags AS (
      SELECT *,
        CASE WHEN is_avg >= 0.9 AND roas < 1.0 THEN 1 ELSE 0 END AS is_overbidding,
        CASE WHEN is_avg <= 0.3 AND roas >= 2.0 THEN 1 ELSE 0 END AS is_budget_constrained
      FROM is_hints
    ),
    is_shares AS (
      SELECT
        SAFE_DIVIDE(SUM(CASE WHEN is_overbidding=1 THEN cost END), NULLIF(SUM(cost),0)) AS pct_spend_overbidding,
        SAFE_DIVIDE(SUM(CASE WHEN is_budget_constrained=1 THEN cost END), NULLIF(SUM(cost),0)) AS pct_spend_budget_constrained
      FROM is_flags
    )
    SELECT 'totals' AS section, TO_JSON_STRING(t) AS json FROM totals t
    UNION ALL
    SELECT 'shares', TO_JSON_STRING(s) FROM shares s
    UNION ALL
    SELECT 'is_shares', TO_JSON_STRING(s2) FROM is_shares s2
    ORDER BY section
    """

    print("\n[Headroom Summary]")
    res = _query_rows_with_fallback(project, sql)
    for r in res:
        print(f"{r['section']}: {r['json']}")

    # Top campaigns in high-CAC and zero-conv buckets
    # High-CAC campaigns
    sql_high = f"""
    WITH kpi AS (
      SELECT DATE(a.date) AS date, a.campaign_id,
             SUM(s.conversions) AS kpi_conversions,
             SUM(s.conversion_value) AS kpi_revenue
      FROM `{project}.{dataset}.ads_conversion_action_stats` s
      JOIN `{project}.{dataset}.ads_campaign_performance` a
        ON DATE(a.date) = DATE(s.date) AND a.campaign_id = s.campaign_id
      WHERE DATE(a.date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
        AND s.conversion_action_id IN ({kpi_ids_csv})
      GROUP BY date, campaign_id
    ),
    cost AS (
      SELECT DATE(date) AS date, campaign_id, SUM(cost_micros)/1e6 AS cost
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      GROUP BY date, campaign_id
    ),
    agg AS (
      SELECT c.campaign_id,
             SUM(c.cost) AS cost,
             SUM(COALESCE(k.kpi_conversions,0)) AS conv,
             SUM(COALESCE(k.kpi_revenue,0)) AS revenue,
             SAFE_DIVIDE(SUM(c.cost), NULLIF(SUM(COALESCE(k.kpi_conversions,0)),0)) AS cac
      FROM cost c LEFT JOIN kpi k USING (date, campaign_id)
      GROUP BY campaign_id
    ),
    stats AS (
      SELECT APPROX_QUANTILES(cac, 100)[OFFSET(50)] AS cac_median
      FROM agg WHERE conv > 0 AND cac IS NOT NULL
    )
    SELECT a.*
    FROM agg a, stats s
    WHERE a.conv > 0 AND a.cac > 1.5*s.cac_median
    ORDER BY a.cost DESC
    LIMIT 10
    """
    print("\n[Top High-CAC Campaigns]")
    for r in _query_rows_with_fallback(project, sql_high):
        print(r)

    # Zero-conversion campaigns
    sql_zero = f"""
    WITH kpi AS (
      SELECT DATE(a.date) AS date, a.campaign_id,
             SUM(s.conversions) AS kpi_conversions,
             SUM(s.conversion_value) AS kpi_revenue
      FROM `{project}.{dataset}.ads_conversion_action_stats` s
      JOIN `{project}.{dataset}.ads_campaign_performance` a
        ON DATE(a.date) = DATE(s.date) AND a.campaign_id = s.campaign_id
      WHERE DATE(a.date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
        AND s.conversion_action_id IN ({kpi_ids_csv})
      GROUP BY date, campaign_id
    ),
    cost AS (
      SELECT DATE(date) AS date, campaign_id, SUM(cost_micros)/1e6 AS cost
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      GROUP BY date, campaign_id
    ),
    agg AS (
      SELECT c.campaign_id,
             SUM(c.cost) AS cost,
             SUM(COALESCE(k.kpi_conversions,0)) AS conv,
             SUM(COALESCE(k.kpi_revenue,0)) AS revenue,
             SAFE_DIVIDE(SUM(c.cost), NULLIF(SUM(COALESCE(k.kpi_conversions,0)),0)) AS cac
      FROM cost c LEFT JOIN kpi k USING (date, campaign_id)
      GROUP BY campaign_id
    )
    SELECT * FROM agg
    WHERE conv = 0 AND cost > 0
    ORDER BY cost DESC
    LIMIT 10
    """
    print("\n[Top Zero-Conversion Campaigns]")
    for r in _query_rows_with_fallback(project, sql_zero):
        print(r)


if __name__ == "__main__":
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET")
    assess(project, dataset)
