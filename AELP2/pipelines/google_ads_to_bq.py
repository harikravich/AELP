#!/usr/bin/env python3
"""
Ingest Google Ads performance into BigQuery (ads_campaign_performance).

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Google Ads env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID (10 digits, no dashes)
- ADC configured for BigQuery: gcloud auth application-default login

Usage:
  python -m AELP2.pipelines.google_ads_to_bq --start 2024-06-01 --end 2024-08-31

Behavior:
- Queries Google Ads API for the date range and writes rows to `${BIGQUERY_TRAINING_DATASET}.ads_campaign_performance`.
- Fails fast with clear errors if missing deps or creds.
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

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
    return f"""
        SELECT
          segments.date,
          campaign.id,
          campaign.name,
          campaign.advertising_channel_type,
          campaign.advertising_channel_sub_type,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc,
          metrics.search_impression_share,
          metrics.search_budget_lost_impression_share,
          metrics.search_rank_lost_impression_share,
          metrics.search_top_impression_share,
          metrics.search_absolute_top_impression_share
        FROM campaign
        WHERE segments.date BETWEEN '{start}' AND '{end}'
    """


def rows_from_response(response) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    redact = os.getenv("AELP2_REDACT_CAMPAIGN_NAMES", "1") == "1"
    for r in response:
        row = {
            "date": r.segments.date,
            "campaign_id": str(r.campaign.id),
            "impressions": int(r.metrics.impressions),
            "clicks": int(r.metrics.clicks),
            "cost_micros": int(r.metrics.cost_micros),
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
            "ctr": float(r.metrics.ctr),
            "avg_cpc_micros": int(r.metrics.average_cpc),
            "impression_share": float(r.metrics.search_impression_share) if r.metrics.search_impression_share is not None else None,
            "lost_is_budget": float(r.metrics.search_budget_lost_impression_share) if r.metrics.search_budget_lost_impression_share is not None else None,
            "lost_is_rank": float(r.metrics.search_rank_lost_impression_share) if r.metrics.search_rank_lost_impression_share is not None else None,
            "top_impression_share": float(r.metrics.search_top_impression_share) if r.metrics.search_top_impression_share is not None else None,
            "abs_top_impression_share": float(r.metrics.search_absolute_top_impression_share) if r.metrics.search_absolute_top_impression_share is not None else None,
        }
        # Channel fields (strings)
        try:
            row["advertising_channel_type"] = str(r.campaign.advertising_channel_type.name)
        except Exception:
            row["advertising_channel_type"] = None
        try:
            row["advertising_channel_sub_type"] = str(r.campaign.advertising_channel_sub_type.name)
        except Exception:
            row["advertising_channel_sub_type"] = None
        name = r.campaign.name
        if redact:
            row["campaign_name"] = None
            row["campaign_name_hash"] = hashlib.sha256((name or "").encode()).hexdigest()
        else:
            row["campaign_name"] = name
            row["campaign_name_hash"] = None
        rows.append(row)
    return rows


def schema(client):
    from google.cloud import bigquery
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("campaign_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("advertising_channel_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("advertising_channel_sub_type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("impressions", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("clicks", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("cost_micros", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("conversion_value", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("avg_cpc_micros", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("impression_share", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("lost_is_budget", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("lost_is_rank", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("top_impression_share", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("abs_top_impression_share", "FLOAT64", mode="NULLABLE"),
    ]


def validate_ads_env() -> None:
    required = [
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET",
        "GOOGLE_ADS_REFRESH_TOKEN",
        "GOOGLE_ADS_CUSTOMER_ID",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing Google Ads env vars: {missing}")
    cid = os.getenv("GOOGLE_ADS_CUSTOMER_ID", "").replace("-", "")
    if not cid.isdigit() or len(cid) != 10:
        raise RuntimeError("GOOGLE_ADS_CUSTOMER_ID must be 10 digits (no dashes)")


def run(start: str, end: str, customer_id: Optional[str] = None) -> None:
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")

    validate_ads_env()

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
    req_cid = (customer_id or os.environ["GOOGLE_ADS_CUSTOMER_ID"]).replace("-", "")
    request.customer_id = req_cid
    request.query = build_gaql(start, end)

    try:
        response = ga_service.search(request=request)
    except GoogleAdsException as e:
        raise RuntimeError(f"Google Ads API error: {e}")

    # Prepare BigQuery
    bq_client = get_bq_client()
    dataset = os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not dataset:
        raise BQIngestionError("BIGQUERY_TRAINING_DATASET env var is required")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset_id = f"{project}.{dataset}"
    ensure_dataset(bq_client, dataset_id)

    table_id = f"{dataset_id}.ads_campaign_performance"
    # Create table if missing; partition by date for efficient backfills/queries
    ensure_table(bq_client, table_id, schema(bq_client), partition_field="date")

    rows = rows_from_response(response)
    if not rows:
        logger.warning("No rows returned for given date range")
        return

    # Stamp customer_id and insert
    for r in rows:
        r["customer_id"] = req_cid
    # Add column if not present
    from google.cloud import bigquery
    table = bq_client.get_table(table_id)
    # Ensure new optional columns exist
    def ensure_col(name: str, typ: str = "STRING"):
        if not any(f.name == name for f in table.schema):
            new_schema = list(table.schema) + [bigquery.SchemaField(name, typ, mode="NULLABLE")]
            table.schema = new_schema
            return True
        return False
    changed = False
    changed |= ensure_col('customer_id', 'STRING')
    changed |= ensure_col('lost_is_budget', 'FLOAT')
    changed |= ensure_col('lost_is_rank', 'FLOAT')
    changed |= ensure_col('top_impression_share', 'FLOAT')
    changed |= ensure_col('abs_top_impression_share', 'FLOAT')
    if changed:
        table = bq_client.update_table(table, ["schema"])  # type: ignore
    insert_rows(bq_client, table_id, rows)
    logger.info(f"Inserted {len(rows)} rows into {table_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    args = p.parse_args()
    # Validate date format early
    for d in [args.start, args.end]:
        datetime.strptime(d, "%Y-%m-%d")
    run(args.start, args.end)
