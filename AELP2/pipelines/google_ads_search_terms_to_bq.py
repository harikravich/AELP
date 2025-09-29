#!/usr/bin/env python3
"""
Load Google Ads search terms into BigQuery (ads_search_terms).
Redacts search term text by default.
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict

from AELP2.pipelines.ads_common import get_ads_client, query_ads_rows, ensure_and_insert, _redact

try:
    from google.cloud import bigquery
    BQ_AVAILABLE = True
except ImportError:
    bigquery = None
    BQ_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def build_gaql(start: str, end: str) -> str:
    return f"""
        SELECT
          segments.date,
          customer.id,
          campaign.id,
          ad_group.id,
          search_term_view.search_term,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc
        FROM search_term_view
        WHERE segments.date BETWEEN '{start}' AND '{end}'
    """


def schema():
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ad_group_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("search_term", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("search_term_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("impressions", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("clicks", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("cost_micros", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("conversions", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("conversion_value", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("ctr", "FLOAT64", mode="NULLABLE"),
        bigquery.SchemaField("avg_cpc_micros", "INT64", mode="NULLABLE"),
    ]


def run(start: str, end: str, customer_id: str):
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    dataset = os.environ["BIGQUERY_TRAINING_DATASET"]
    redact_all = os.getenv("AELP2_REDACT_TEXT", "1") == "1"

    client = get_ads_client(os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"))
    rows_iter = query_ads_rows(client, customer_id, build_gaql(start, end))

    out_rows: List[Dict] = []
    for r in rows_iter:
        st = r.search_term_view.search_term
        rn = {
            "date": r.segments.date,
            "customer_id": r.customer.id,
            "campaign_id": str(r.campaign.id),
            "ad_group_id": str(r.ad_group.id),
            "impressions": int(r.metrics.impressions),
            "clicks": int(r.metrics.clicks),
            "cost_micros": int(r.metrics.cost_micros),
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
            "ctr": float(r.metrics.ctr),
            "avg_cpc_micros": int(r.metrics.average_cpc),
        }
        s = _redact(st, enabled=redact_all)
        rn["search_term"] = s["text"]
        rn["search_term_hash"] = s["hash"]
        out_rows.append(rn)

    ensure_and_insert(project, dataset, "ads_search_terms", schema(), out_rows, partition_field="date")
    logger.info(f"Inserted {len(out_rows)} rows into {dataset}.ads_search_terms for customer {customer_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--customer", required=True, help="Customer ID (10 digits)")
    args = p.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    run(args.start, args.end, customer_id=args.customer)

