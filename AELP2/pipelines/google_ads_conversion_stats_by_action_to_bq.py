#!/usr/bin/env python3
"""
Load Google Ads conversion stats by conversion_action into BigQuery
(ads_conversion_action_stats).

Fields: date, customer_id, campaign_id, conversion_action_id, conversion_action_name (redacted),
conversions, conversion_value, cost_micros

Usage:
  python -m AELP2.pipelines.google_ads_conversion_stats_by_action_to_bq --start YYYY-MM-DD --end YYYY-MM-DD [--customer 1234567890]
"""

import os
import argparse
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional

from AELP2.core.ingestion.bq_loader import (
    get_bq_client, ensure_dataset, ensure_table, insert_rows, BQIngestionError
)

try:
    from google.ads.googleads.client import GoogleAdsClient
    from google.ads.googleads.errors import GoogleAdsException
    ADS_AVAILABLE = True
except ImportError as e:
    GoogleAdsClient = None
    GoogleAdsException = Exception
    ADS_AVAILABLE = False
    ADS_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def build_gaql(start: str, end: str) -> str:
    """Build a valid GAQL for conversion stats by action.

    Use campaign as the base resource and segments.conversion_action as the
    identifier, which is allowed and returns the resource name that we parse
    to a numeric id.
    """
    return f"""
        SELECT
          segments.date,
          customer.id,
          campaign.id,
          segments.conversion_action,
          metrics.conversions,
          metrics.conversions_value
        FROM campaign
        WHERE segments.date BETWEEN '{start}' AND '{end}'
          AND segments.conversion_action IS NOT NULL
    """


def schema():
    from google.cloud import bigquery
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("conversion_action_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("conversion_action_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("conversion_action_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("conversion_value", "FLOAT64", mode="NULLABLE"),
    ]


def run(start: str, end: str, customer_id: Optional[str] = None) -> None:
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")

    # Build Google Ads client from env
    cfg = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "use_proto_plus": True,
    }
    login_cid = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login_cid:
        cfg["login_customer_id"] = login_cid.replace("-", "")
    client = GoogleAdsClient.load_from_dict(cfg)

    ga_service = client.get_service("GoogleAdsService")
    request = client.get_type("SearchGoogleAdsRequest")
    req_cid = (customer_id or os.environ.get("GOOGLE_ADS_CUSTOMER_ID", "")).replace("-", "")
    if not req_cid:
        raise RuntimeError("Provide --customer or set GOOGLE_ADS_CUSTOMER_ID")
    request.customer_id = req_cid
    request.query = build_gaql(start, end)
    logger.warning(f"GAQL (conversion_action_stats):\n{request.query}")

    try:
        response = ga_service.search(request=request)
    except GoogleAdsException as e:
        raise RuntimeError(f"Google Ads API error: {e}")

    redact = os.getenv("AELP2_REDACT_TEXT", "1") == "1"
    rows: List[Dict] = []
    for r in response:
        # segments.conversion_action is a resource name like
        # "customers/{cid}/conversionActions/{id}" — parse the ID
        res_name = str(r.segments.conversion_action)
        ca_id = res_name.split('/')[-1] if res_name else None
        # We no longer fetch the name directly in this GAQL; store redacted/None
        n = None
        h = None
        if not redact:
            n = None
            h = None
        rows.append({
            "date": r.segments.date,
            "customer_id": r.customer.id,
            "campaign_id": str(r.campaign.id),
            "conversion_action_id": ca_id,
            "conversion_action_name": n,
            "conversion_action_name_hash": h,
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
        })

    bq = get_bq_client()
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    dataset = os.environ["BIGQUERY_TRAINING_DATASET"]
    table_id = f"{project}.{dataset}.ads_conversion_action_stats"
    ensure_table(bq, table_id, schema(), partition_field="date")
    # If an older table exists with cost_micros, we will write without that field; BigQuery accepts missing nullable fields.
    if rows:
        insert_rows(bq, table_id, rows)
        logger.info(f"Inserted {len(rows)} rows into {table_id}")
    else:
        logger.info("No conversion action stats rows returned for window")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--customer", required=False)
    args = p.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    run(args.start, args.end, customer_id=args.customer)
