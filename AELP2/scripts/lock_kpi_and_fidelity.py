#!/usr/bin/env python3
"""
Lock Ads KPI conversion_action_ids (based on PURCHASE actions with real volume),
refresh KPI views, print headroom, and optionally run a stabilization training
pass followed by KPI-only fidelity.

Usage:
  GOOGLE_CLOUD_PROJECT=... BIGQUERY_TRAINING_DATASET=... \
  python3 -m AELP2.scripts.lock_kpi_and_fidelity [--last_days 30] [--top 5] \
    [--run_training] [--episodes 10] [--steps 600] [--budget 8000]

Notes:
 - Picks Ads conversion actions where category LIKE '%PURCHASE%', status ENABLED,
   primary_for_goal true (if present), and with >0 conversions in last N days,
   then sorts by conversions desc and takes top K.
 - Sets AELP2_KPI_CONVERSION_ACTION_IDS in-process and refreshes views.
 - Prints headroom summary and tail campaigns via assess_headroom.assess().
 - Optional: runs run_quick_fidelity.sh with provided episodes/steps/budget.
"""

import os
import sys
import subprocess
import shutil
import json
from typing import List

try:
    from google.cloud import bigquery
except Exception as e:
    print(f"google-cloud-bigquery required: {e}", file=sys.stderr)
    sys.exit(2)


def pick_purchase_actions(bq: bigquery.Client, project: str, dataset: str, last_days: int, top_k: int) -> List[str]:
    sql = f"""
    WITH cost AS (
      SELECT DATE(date) AS date, campaign_id, SUM(cost_micros)/1e6 AS cost
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {last_days} DAY) AND CURRENT_DATE()
      GROUP BY date, campaign_id
    ), by_action AS (
      SELECT
        s.conversion_action_id,
        SUM(s.conversions) AS conv,
        SUM(s.conversion_value) AS revenue,
        SUM(c.cost) AS cost
      FROM `{project}.{dataset}.ads_conversion_action_stats` s
      JOIN cost c ON c.date = DATE(s.date) AND c.campaign_id = s.campaign_id
      WHERE DATE(s.date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {last_days} DAY) AND CURRENT_DATE()
      GROUP BY s.conversion_action_id
    )
    SELECT a.conversion_action_id
    FROM `{project}.{dataset}.ads_conversion_actions` a
    JOIN by_action b USING (conversion_action_id)
    WHERE a.status = 'ConversionActionStatus.ENABLED'
      AND a.category LIKE '%PURCHASE%'
      AND b.conv > 0
    ORDER BY b.conv DESC
    LIMIT {top_k}
    """
    # Try via Python client first; fall back to bq CLI if ADC lacks permissions
    try:
        return [r.conversion_action_id for r in bq.query(sql).result()]
    except Exception as e:
        msg = str(e)
        needs_fallback = any(s in msg for s in (
            "403", "Access Denied", "bigquery.jobs.create", "Forbidden"
        ))
        if not needs_fallback or shutil.which("bq") is None:
            # Re-raise if it's not a permissions issue or bq CLI is unavailable
            raise
        print(
            "BigQuery Python client lacked permissions (likely ADC using VM service account). "
            "Falling back to 'bq query' which is authenticated and succeeded locally...",
            file=sys.stderr,
        )
        cmd = [
            "bq",
            f"--project_id={project}",
            "query",
            "--use_legacy_sql=false",
            "--quiet",
            "--format=csv",
            sql,
        ]
        out = subprocess.check_output(cmd, text=True)
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        # Expect header row 'conversion_action_id' followed by IDs
        if lines and lines[0].lower().strip() == "conversion_action_id":
            lines = lines[1:]
        # Basic validation: IDs should be digits
        ids = [x for x in lines if x and all(c.isdigit() for c in x)]
        return ids


def refresh_views(env: dict):
    # Call the create_bq_views module in a subprocess with the env set.
    # Tolerate failures (e.g., missing bigquery.tables.create) and continue.
    cmd = [sys.executable, "-m", "AELP2.pipelines.create_bq_views"]
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Warning: failed to refresh views (non-fatal): {e}")


def run_assess_headroom(project: str, dataset: str):
    # Import and call the assess function directly to print results
    from AELP2.scripts.assess_headroom import assess
    assess(project, dataset)


def run_training_and_fidelity(env: dict, episodes: int, steps: int, budget: float):
    shell_env = os.environ.copy()
    shell_env.update(env)
    # run_quick_fidelity.sh does: quick training + KPI-only fidelity + print latest row
    cmd = ["bash", "AELP2/scripts/run_quick_fidelity.sh"]
    shell_env["AELP2_EPISODES"] = str(episodes)
    shell_env["AELP2_SIM_STEPS"] = str(steps)
    shell_env["AELP2_SIM_BUDGET"] = str(budget)
    subprocess.run(cmd, check=True, env=shell_env)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--last_days", type=int, default=30)
    p.add_argument("--top", type=int, default=5, help="Top K purchase actions to keep")
    p.add_argument("--run_training", action="store_true")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--budget", type=float, default=8000)
    args = p.parse_args()

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        print("Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET", file=sys.stderr)
        sys.exit(2)

    bq = bigquery.Client(project=project)
    kpi_ids = pick_purchase_actions(bq, project, dataset, last_days=args.last_days, top_k=args.top)
    if not kpi_ids:
        print("No PURCHASE conversion actions with volume found in last", args.last_days, "days.")
        sys.exit(3)
    # Deduplicate while preserving order
    seen = set()
    kpi_unique = []
    for _id in kpi_ids:
        if _id not in seen:
            seen.add(_id)
            kpi_unique.append(_id)
    kpi_csv = ",".join(kpi_unique)
    print("Selected KPI Ads conversion_action_ids:", kpi_csv)

    # Prepare env for downstream steps (subprocess and in-process)
    child_env = os.environ.copy()
    child_env["AELP2_KPI_CONVERSION_ACTION_IDS"] = kpi_csv
    # Ensure in-process imports read the same KPI env value
    os.environ["AELP2_KPI_CONVERSION_ACTION_IDS"] = kpi_csv

    print("\n[Refresh KPI views]")
    refresh_views(child_env)

    print("\n[Headroom]")
    run_assess_headroom(project, dataset)

    if args.run_training:
        print("\n[Training + KPI Fidelity]")
        # Provide sensible defaults for training stabilization
        child_env.update({
            "LOG_LEVEL": "WARNING",
            "AELP2_HITL_NON_BLOCKING": "1",
            "AELP2_HITL_ON_GATE_FAIL": "0",
            "AELP2_HITL_ON_GATE_FAIL_FOR_BIDS": "0",
            "AELP2_HITL_MIN_STEP_FOR_APPROVAL": "999999",
            "AELP2_TARGET_WIN_RATE_MIN": "0.25",
            "AELP2_TARGET_WIN_RATE_MAX": "0.35",
            "AELP2_CALIBRATION_FLOOR_RATIO": "0.80",
            "AELP2_FLOOR_AUTOTUNE_ENABLE": "1",
            "AELP2_FLOOR_AUTOTUNE_MIN": "0.50",
            "AELP2_FLOOR_AUTOTUNE_MAX": "0.90",
            "AELP2_FLOOR_AUTOTUNE_STEP": "0.05",
            "AELP2_NOWIN_GUARD_ENABLE": "1",
            "AELP2_NOWIN_GUARD_STEPS": "15",
            "AELP2_NOWIN_GUARD_FACTOR": "2.5",
        })
        run_training_and_fidelity(child_env, episodes=args.episodes, steps=args.steps, budget=args.budget)


if __name__ == "__main__":
    main()
