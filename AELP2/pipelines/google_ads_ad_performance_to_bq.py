#!/usr/bin/env python3
"""
Load Google Ads ad performance into BigQuery (ads_ad_performance).

Fields include ad id, ad group, campaign, (redacted) ad name/headline, impressions, clicks, cost, conversions, value.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Ads env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID

Usage:
  python -m AELP2.pipelines.google_ads_ad_performance_to_bq --start 2024-07-01 --end 2024-07-31 --customer 1234567890
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict
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
          customer.id,
          campaign.id,
          campaign.name,
          ad_group.id,
          ad_group.name,
          ad_group_ad.ad.id,
          ad_group_ad.ad.name,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc
        FROM ad_group_ad
        WHERE segments.date BETWEEN '{start}' AND '{end}'
          AND ad_group_ad.status != 'REMOVED'
    """


def schema():
    from google.cloud import bigquery
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("campaign_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_group_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ad_group_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_group_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ad_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("impressions", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("clicks", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("cost_micros", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("conversion_value", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("avg_cpc_micros", "INT64", mode="NULLABLE"),
    ]


def run(start: str, end: str, customer_id: str | None = None):
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")

    # Build client
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
    if not customer_id:
        env_cid = os.environ.get("GOOGLE_ADS_CUSTOMER_ID")
        if not env_cid:
            raise RuntimeError("Missing --customer and GOOGLE_ADS_CUSTOMER_ID env var")
        customer_id = env_cid
    req_cid = customer_id.replace("-", "")
    request.customer_id = req_cid
    request.query = build_gaql(start, end)

    try:
        response = ga_service.search(request=request)
    except GoogleAdsException as e:
        raise RuntimeError(f"Google Ads API error: {e}")

    redact_all = os.getenv("AELP2_REDACT_TEXT", "1") == "1"
    out_rows: List[Dict] = []
    for r in response:
        camp_name = r.campaign.name
        adg_name = r.ad_group.name
        ad_name = r.ad_group_ad.ad.name
        rn = {
            "date": r.segments.date,
            "customer_id": r.customer.id,
            "campaign_id": str(r.campaign.id),
            "ad_group_id": str(r.ad_group.id),
            "ad_id": str(r.ad_group_ad.ad.id),
            "impressions": int(r.metrics.impressions),
            "clicks": int(r.metrics.clicks),
            "cost_micros": int(r.metrics.cost_micros),
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
            "ctr": float(r.metrics.ctr),
            "avg_cpc_micros": int(r.metrics.average_cpc),
        }
        # Redaction
        def _redact(text: str):
            if not redact_all:
                return text, None
            return None, hashlib.sha256((text or "").encode()).hexdigest()

        rn["campaign_name"], rn["campaign_name_hash"] = _redact(camp_name)
        rn["ad_group_name"], rn["ad_group_name_hash"] = _redact(adg_name)
        rn["ad_name"], rn["ad_name_hash"] = _redact(ad_name)
        out_rows.append(rn)

    bq = get_bq_client()
    dataset = os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not dataset:
        raise BQIngestionError("BIGQUERY_TRAINING_DATASET env var is required")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset_id = f"{project}.{dataset}"
    ensure_dataset(bq, dataset_id)
    table_id = f"{dataset_id}.ads_ad_performance"
    ensure_table(bq, table_id, schema(), partition_field="date")
    if out_rows:
        insert_rows(bq, table_id, out_rows)
        logger.info(f"Inserted {len(out_rows)} rows into {table_id}")
    else:
        logger.warning("No ad performance rows returned for given range")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--customer", required=False, help="Customer ID (10 digits); defaults to GOOGLE_ADS_CUSTOMER_ID")
    args = p.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    run(args.start, args.end, customer_id=args.customer)
