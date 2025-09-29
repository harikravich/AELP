#!/usr/bin/env python3
"""
Ingest GA4 aggregates into BigQuery (ga4_aggregates).

Note: For raw events, set up native GA4→BigQuery export. This script uses
the GA4 Data API for aggregated daily metrics to bootstrap calibration/monitoring.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, GA4_PROPERTY_ID
- ADC configured for BigQuery and GA4 Data API access

Usage:
  python -m AELP2.pipelines.ga4_to_bq --start 2024-06-01 --end 2024-08-31
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict

from AELP2.core.ingestion.bq_loader import (
    get_bq_client, ensure_dataset, ensure_table, insert_rows, BQIngestionError
)

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google.auth.transport.requests import Request as OAuthRequest
    import google.auth
    from google.api_core import exceptions as ga_exceptions
    GA4_AVAILABLE = True
except ImportError as e:
    BetaAnalyticsDataClient = None
    DateRange = Dimension = Metric = RunReportRequest = None
    OAuthCredentials = None
    ServiceAccountCredentials = None
    OAuthRequest = None
    google = None
    GA4_AVAILABLE = False
    GA4_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

GA4_SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]


def _get_ga4_client() -> "BetaAnalyticsDataClient":
    """Construct a GA4 client with proper scopes.

    Preference order:
      1) Explicit OAuth refresh token path via GA4_OAUTH_* envs
      2) Service Account JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS
      3) Application Default Credentials with explicit scopes
    """
    # 1) OAuth Refresh Token path
    rt = os.getenv("GA4_OAUTH_REFRESH_TOKEN")
    cid = os.getenv("GA4_OAUTH_CLIENT_ID")
    cs = os.getenv("GA4_OAUTH_CLIENT_SECRET")
    if rt and cid and cs and OAuthCredentials and OAuthRequest:
        creds = OAuthCredentials(
            token=None,
            refresh_token=rt,
            client_id=cid,
            client_secret=cs,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=GA4_SCOPES,
        )
        creds.refresh(OAuthRequest())
        logger.info("GA4 auth: using OAuth refresh token credentials (analytics.readonly)")
        return BetaAnalyticsDataClient(credentials=creds)

    # 2) Service Account JSON with explicit scopes
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path:
        # Expand ~ in case the env var was set with a tilde path
        sa_path = os.path.expanduser(sa_path)
    if sa_path and ServiceAccountCredentials and os.path.isfile(sa_path):
        sa_creds = ServiceAccountCredentials.from_service_account_file(sa_path, scopes=GA4_SCOPES)
        logger.info("GA4 auth: using service account JSON from GOOGLE_APPLICATION_CREDENTIALS with analytics.readonly scope")
        return BetaAnalyticsDataClient(credentials=sa_creds)

    # 3) ADC with explicit scopes (works for SA ADC; user ADC must include these scopes)
    if google is not None:
        try:
            adc_creds, _ = google.auth.default(scopes=GA4_SCOPES)
            logger.info("GA4 auth: using Application Default Credentials with analytics.readonly scope")
            return BetaAnalyticsDataClient(credentials=adc_creds)
        except Exception as e:
            logger.warning(f"GA4 auth: ADC with scopes failed ({e}); falling back to default client")

    # Fallback to library defaults (may fail if ADC lacks scopes)
    logger.warning("GA4 auth: falling back to BetaAnalyticsDataClient() with library defaults; if you see 'ACCESS_TOKEN_SCOPE_INSUFFICIENT', provide GA4_OAUTH_* or a service account JSON")
    return BetaAnalyticsDataClient()


def run(start: str, end: str, dry_run: bool = False) -> None:
    if dry_run or os.getenv('AELP2_DRY_RUN', '0') == '1':
        print(f"[dry_run] Would fetch GA4 aggregates for {start}..{end} and write to ga4_aggregates; requires GA4_PROPERTY_ID + creds.")
        return
    if not GA4_AVAILABLE:
        raise RuntimeError(f"google-analytics-data not installed: {GA4_IMPORT_ERROR}")

    prop = os.getenv("GA4_PROPERTY_ID")
    if not prop:
        raise RuntimeError("GA4_PROPERTY_ID env var is required (format: properties/<id>)")

    # Build a GA4 client with proper scopes
    client = _get_ga4_client()

    request = RunReportRequest(
        property=prop,
        date_ranges=[DateRange(start_date=start, end_date=end)],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="deviceCategory"),
            Dimension(name="defaultChannelGroup"),
        ],
        metrics=[
            Metric(name="sessions"),
            Metric(name="conversions"),
            Metric(name="totalUsers"),
        ],
    )

    try:
        response = client.run_report(request)
    except Exception as e:  # Provide actionable diagnostics on common auth failures
        msg = str(e)
        if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in msg or "insufficient authentication scopes" in msg:
            raise RuntimeError(
                "GA4 RunReport failed due to insufficient OAuth scopes. "
                "Fix by either: (a) setting GOOGLE_APPLICATION_CREDENTIALS to a Service Account JSON that has GA4 property access, or "
                "(b) providing GA4_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN with the 'analytics.readonly' scope."
            ) from e
        raise

    rows: List[Dict] = []
    for r in response.rows:
        rows.append({
            "date": r.dimension_values[0].value,
            "device_category": r.dimension_values[1].value,
            "default_channel_group": r.dimension_values[2].value,
            "sessions": int(r.metric_values[0].value or 0),
            "conversions": int(float(r.metric_values[1].value or 0)),
            "users": int(r.metric_values[2].value or 0),
        })

    # Prepare BigQuery
    bq_client = get_bq_client()
    dataset = os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not dataset:
        raise BQIngestionError("BIGQUERY_TRAINING_DATASET env var is required")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset_id = f"{project}.{dataset}"
    ensure_dataset(bq_client, dataset_id)

    # Create table and insert
    from google.cloud import bigquery
    table_id = f"{dataset_id}.ga4_aggregates"
    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("device_category", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("default_channel_group", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("sessions", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("conversions", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("users", "INT64", mode="NULLABLE"),
    ]
    ensure_table(bq_client, table_id, schema, partition_field="date")

    if not rows:
        logger.warning("No GA4 rows returned for given date range")
        return

    # Convert date string to YYYY-MM-DD for DATE column
    for row in rows:
        # GA4 date comes as YYYYMMDD
        d = row["date"]
        row["date"] = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    insert_rows(bq_client, table_id, rows)
    logger.info(f"Inserted {len(rows)} rows into {table_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    # Validate date format early
    for d in [args.start, args.end]:
        datetime.strptime(d, "%Y-%m-%d")
    run(args.start, args.end, dry_run=args.dry_run)
