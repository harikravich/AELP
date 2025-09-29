"""
Shared utilities for Google Ads → BigQuery loaders.

Strictness:
- Read env for all configuration (no hardcoding).
- Redact free-text fields by default (hash) to avoid storing raw campaign/ad/search strings.
- Fail-fast with clear errors if credentials missing.
"""

import os
import hashlib
import logging
from typing import Dict, Iterable, List

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


def _redact(value: str, enabled: bool = True) -> Dict[str, str]:
    if not value:
        return {"text": None, "hash": None}
    if not enabled:
        return {"text": value, "hash": None}
    return {"text": None, "hash": hashlib.sha256(value.encode()).hexdigest()}


def get_ads_client(login_cid: str | None = None) -> GoogleAdsClient:
    if not ADS_AVAILABLE:
        raise RuntimeError(f"google-ads not installed: {ADS_IMPORT_ERROR}")
    required = [
        "GOOGLE_ADS_DEVELOPER_TOKEN",
        "GOOGLE_ADS_CLIENT_ID",
        "GOOGLE_ADS_CLIENT_SECRET",
        "GOOGLE_ADS_REFRESH_TOKEN",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing Google Ads env vars: {missing}")
    cfg = {
        "developer_token": os.environ["GOOGLE_ADS_DEVELOPER_TOKEN"],
        "client_id": os.environ["GOOGLE_ADS_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_ADS_CLIENT_SECRET"],
        "refresh_token": os.environ["GOOGLE_ADS_REFRESH_TOKEN"],
        "use_proto_plus": True,
    }
    login = login_cid or os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login:
        cfg["login_customer_id"] = login.replace("-", "")
    return GoogleAdsClient.load_from_dict(cfg)


def query_ads_rows(client: GoogleAdsClient, customer_id: str, gaql: str) -> Iterable:
    ga_service = client.get_service("GoogleAdsService")
    request = client.get_type("SearchGoogleAdsRequest")
    request.customer_id = customer_id.replace("-", "")
    request.query = gaql
    return ga_service.search(request=request)


def ensure_and_insert(project: str, dataset: str, table: str, schema, rows: List[Dict], partition_field: str | None = None) -> None:
    bq = get_bq_client()
    ensure_dataset(bq, f"{project}.{dataset}")
    ensure_table(bq, f"{project}.{dataset}.{table}", schema, partition_field=partition_field)
    insert_rows(bq, f"{project}.{dataset}.{table}", rows)


__all__ = [
    "_redact",
    "get_ads_client",
    "query_ads_rows",
    "ensure_and_insert",
]

