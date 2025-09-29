#!/usr/bin/env python3
"""
Basic data quality and freshness checks for core BigQuery tables.

Checks:
- Existence of required tables
- Freshness (rows within last N days) for ads_campaign_performance and training_episodes
- Null ratios for key fields

Exit codes: 0 ok, 1 warnings, 2 failures.
"""

import os
import logging
from datetime import datetime, timedelta

try:
    from google.cloud import bigquery
except Exception as e:
    raise SystemExit(f"google-cloud-bigquery required: {e}")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def main() -> int:
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset = os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        print("Missing GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET")
        return 2
    client = bigquery.Client(project=project)
    ds = f"{project}.{dataset}"
    required = [
        "ads_campaign_performance",
        "training_episodes",
    ]
    status = 0
    # Existence
    for t in required:
        try:
            client.get_table(f"{ds}.{t}")
        except Exception:
            print(f"FAIL: Missing table {ds}.{t}")
            status = 2

    # Freshness: last 14 days
    freshness_checks = {
        "ads_campaign_performance": "SELECT MAX(DATE(date)) AS d FROM `{ds}.ads_campaign_performance`".format(ds=ds),
        "training_episodes": "SELECT MAX(DATE(timestamp)) AS d FROM `{ds}.training_episodes`".format(ds=ds),
    }
    horizon = datetime.utcnow().date() - timedelta(days=14)
    for name, sql in freshness_checks.items():
        try:
            rows = list(client.query(sql).result())
            if not rows or rows[0][0] is None:
                print(f"WARN: No data in {name}")
                status = max(status, 1)
            else:
                last = rows[0][0]
                if last < horizon:
                    print(f"WARN: {name} last date {last} older than 14 days")
                    status = max(status, 1)
        except Exception as e:
            print(f"WARN: Freshness check failed for {name}: {e}")
            status = max(status, 1)

    # Null ratios example for ads campaign
    try:
        sql = f"""
            SELECT
              SAFE_DIVIDE(SUM(CASE WHEN impressions IS NULL THEN 1 ELSE 0 END), COUNT(*)) AS null_impr,
              SAFE_DIVIDE(SUM(CASE WHEN clicks IS NULL THEN 1 ELSE 0 END), COUNT(*)) AS null_clicks
            FROM `{ds}.ads_campaign_performance`
        """
        r = list(client.query(sql).result())
        if r:
            m = dict(r[0])
            if (m.get('null_impr') or 0) > 0.01 or (m.get('null_clicks') or 0) > 0.01:
                print(f"WARN: High null ratio in ads_campaign_performance: {m}")
                status = max(status, 1)
    except Exception as e:
        print(f"WARN: Null ratio check failed: {e}")
        status = max(status, 1)

    print("DQ status:", "OK" if status == 0 else ("WARN" if status == 1 else "FAIL"))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

