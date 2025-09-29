#!/usr/bin/env python3
"""
Load Google Ads conversion actions into BigQuery (ads_conversion_actions).

This loader ingests conversion action definitions (id, name, category, type, status, primary_for_goal)
for use in attribution mapping and downstream analytics. Names are redacted by default.

Requirements:
- Env: GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
- Ads env: GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID (10 digits)

Usage:
  python -m AELP2.pipelines.google_ads_conversion_actions_to_bq
"""

import os
import logging
import hashlib
from typing import List, Dict

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


def schema():
    from google.cloud import bigquery
    return [
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("conversion_action_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("category", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("type", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("include_in_conversions_metric", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("primary_for_goal", "BOOL", mode="NULLABLE"),
        bigquery.SchemaField("loaded_at", "TIMESTAMP", mode="REQUIRED"),
    ]


def run(customer_id: str = None) -> int:
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")

    validate_ads_env()

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

    req_cid = (customer_id or os.environ["GOOGLE_ADS_CUSTOMER_ID"]).replace("-", "")

    ga_service = client.get_service("GoogleAdsService")
    request = client.get_type("SearchGoogleAdsRequest")
    request.customer_id = req_cid
    request.query = """
        SELECT
          conversion_action.id,
          conversion_action.name,
          conversion_action.category,
          conversion_action.type,
          conversion_action.status,
          conversion_action.include_in_conversions_metric,
          conversion_action.primary_for_goal
        FROM conversion_action
      """
    try:
        response = ga_service.search(request=request)
    except GoogleAdsException as e:
        raise RuntimeError(f"Google Ads API error: {e}")

    redact = os.getenv("AELP2_REDACT_TEXT", "1") == "1"
    rows: List[Dict] = []
    from datetime import datetime as _dt
    loaded_at = _dt.utcnow().isoformat()
    for r in response:
        name = r.conversion_action.name
        if redact:
            name_hash = hashlib.sha256((name or "").encode()).hexdigest()
            name_val = None
        else:
            name_hash = None
            name_val = name
        rows.append({
            "customer_id": req_cid,
            "conversion_action_id": str(r.conversion_action.id),
            "name": name_val,
            "name_hash": name_hash,
            "category": str(r.conversion_action.category),
            "type": str(r.conversion_action.type),
            "status": str(r.conversion_action.status),
            "include_in_conversions_metric": bool(getattr(r.conversion_action, 'include_in_conversions_metric', True)),
            "primary_for_goal": bool(r.conversion_action.primary_for_goal),
            "loaded_at": loaded_at,
        })

    bq = get_bq_client()
    dataset = os.environ.get("BIGQUERY_TRAINING_DATASET")
    if not dataset:
        raise BQIngestionError("BIGQUERY_TRAINING_DATASET env var is required")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    dataset_id = f"{project}.{dataset}"
    ensure_dataset(bq, dataset_id)
    table_id = f"{dataset_id}.ads_conversion_actions"
    ensure_table(bq, table_id, schema(), partition_field=None)
    if rows:
        insert_rows(bq, table_id, rows)
        logger.info(f"Inserted {len(rows)} conversion actions into {table_id}")
    else:
        logger.warning("No conversion actions returned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
