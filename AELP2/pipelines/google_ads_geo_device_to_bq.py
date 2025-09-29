#!/usr/bin/env python3
"""
Load Google Ads device-level performance into BigQuery (ads_geo_device_performance).

Captures performance by device for campaign rollups.
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List, Dict

from AELP2.pipelines.ads_common import get_ads_client, query_ads_rows, ensure_and_insert

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
          segments.device,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc
        FROM campaign
        WHERE segments.date BETWEEN '{start}' AND '{end}'
    """


def schema():
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("device", "STRING", mode="NULLABLE"),
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

    client = get_ads_client(os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID"))
    rows_iter = query_ads_rows(client, customer_id, build_gaql(start, end))

    out_rows: List[Dict] = []
    for r in rows_iter:
        rn = {
            "date": r.segments.date,
            "customer_id": r.customer.id,
            "campaign_id": str(r.campaign.id),
            "device": str(r.segments.device),
            "impressions": int(r.metrics.impressions),
            "clicks": int(r.metrics.clicks),
            "cost_micros": int(r.metrics.cost_micros),
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
            "ctr": float(r.metrics.ctr),
            "avg_cpc_micros": int(r.metrics.average_cpc),
        }
        out_rows.append(rn)

    ensure_and_insert(project, dataset, "ads_geo_device_performance", schema(), out_rows, partition_field="date")
    logger.info(f"Inserted {len(out_rows)} rows into {dataset}.ads_geo_device_performance for customer {customer_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--customer", required=True, help="Customer ID (10 digits)")
    args = p.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    run(args.start, args.end, customer_id=args.customer)

