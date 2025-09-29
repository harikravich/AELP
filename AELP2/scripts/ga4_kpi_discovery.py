#!/usr/bin/env python3
"""
GA4 KPI Discovery

Discovers GA4 conversion events and summarizes last-N-days metrics to help
calibrate the true KPI (e.g., sign_up, generate_lead, purchase).

Outputs:
- Top GA4 events by conversions with: conversions, event_count, users,
  event_value, purchase_revenue
- Optional write to BigQuery table `<project>.<dataset>.ga4_conversion_events_summary`

Auth:
- Uses either GA4_OAUTH_* (refresh-token) or GOOGLE_APPLICATION_CREDENTIALS
  service account JSON, same as ga4_to_bq.py. Requires GA4 Data API access.

Env:
  GA4_PROPERTY_ID (required, format: properties/<id>)
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET (optional for BQ write)

Usage:
  python3 -m AELP2.scripts.ga4_kpi_discovery --last_days 30 [--write_bq]
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google.auth.transport.requests import Request as OAuthRequest
    import google.auth
    GA4_AVAILABLE = True
except Exception as e:
    BetaAnalyticsDataClient = None
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
        logger.info("GA4 auth: OAuth refresh token credentials")
        return BetaAnalyticsDataClient(credentials=creds)
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and ServiceAccountCredentials and os.path.isfile(os.path.expanduser(sa_path)):
        sa_creds = ServiceAccountCredentials.from_service_account_file(os.path.expanduser(sa_path), scopes=GA4_SCOPES)
        logger.info("GA4 auth: service account JSON")
        return BetaAnalyticsDataClient(credentials=sa_creds)
    if google is not None:
        adc_creds, _ = google.auth.default(scopes=GA4_SCOPES)
        logger.info("GA4 auth: Application Default Credentials")
        return BetaAnalyticsDataClient(credentials=adc_creds)
    return BetaAnalyticsDataClient()


def discover(property_id: str, start: str, end: str) -> List[Dict]:
    if not GA4_AVAILABLE:
        raise RuntimeError(f"google-analytics-data not installed: {GA4_IMPORT_ERROR}")
    client = _get_ga4_client()
    req = RunReportRequest(
        property=property_id,
        date_ranges=[DateRange(start_date=start, end_date=end)],
        dimensions=[Dimension(name="eventName")],
        metrics=[
            Metric(name="conversions"),
            Metric(name="eventCount"),
            Metric(name="totalUsers"),
            Metric(name="eventValue"),
            Metric(name="purchaseRevenue"),
        ],
    )
    resp = client.run_report(req)
    out: List[Dict] = []
    for r in resp.rows:
        out.append({
            "event_name": r.dimension_values[0].value,
            "conversions": float(r.metric_values[0].value or 0),
            "event_count": int(r.metric_values[1].value or 0),
            "users": int(r.metric_values[2].value or 0),
            "event_value": float(r.metric_values[3].value or 0),
            "purchase_revenue": float(r.metric_values[4].value or 0),
        })
    # Sort by conversions desc
    out.sort(key=lambda x: x["conversions"], reverse=True)
    return out


def maybe_write_bq(rows: List[Dict], start: str, end: str):
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET")
    if not project or not dataset:
        logger.info("BQ write skipped (GOOGLE_CLOUD_PROJECT or BIGQUERY_TRAINING_DATASET not set)")
        return
    try:
        from google.cloud import bigquery
    except Exception as e:
        logger.warning(f"BQ write skipped: {e}")
        return
    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.ga4_conversion_events_summary"
    schema = [
        bigquery.SchemaField("computed_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("window_start", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("window_end", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("event_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("conversions", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("event_count", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("users", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("event_value", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("purchase_revenue", "FLOAT", mode="NULLABLE"),
    ]
    # Ensure table
    try:
        bq.get_table(table_id)
    except Exception:
        table = bigquery.Table(table_id, schema=schema)
        bq.create_table(table)
    now = datetime.utcnow().isoformat()
    payload = []
    for r in rows:
        rr = dict(r)
        rr["computed_at"] = now
        rr["window_start"] = start
        rr["window_end"] = end
        payload.append(rr)
    if payload:
        bq.insert_rows_json(table_id, payload)
        logger.info(f"Wrote {len(payload)} rows to {table_id}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--last_days", type=int, default=30)
    p.add_argument("--write_bq", action="store_true")
    args = p.parse_args()

    prop = os.getenv("GA4_PROPERTY_ID")
    if not prop:
        raise SystemExit("Set GA4_PROPERTY_ID env (format: properties/<id>)")

    end = datetime.utcnow().date()
    start = (end - timedelta(days=args.last_days)).isoformat()
    end_s = end.isoformat()

    rows = discover(prop, start, end_s)
    print("\n[GA4 Conversion Events (last", args.last_days, "days)]")
    print("event_name, conversions, event_count, users, event_value, purchase_revenue")
    for r in rows[:50]:
        print(f"{r['event_name']}, {r['conversions']}, {r['event_count']}, {r['users']}, {r['event_value']}, {r['purchase_revenue']}")

    if args.write_bq:
        maybe_write_bq(rows, start, end_s)


if __name__ == "__main__":
    main()

