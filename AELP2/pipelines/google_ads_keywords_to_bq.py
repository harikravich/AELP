#!/usr/bin/env python3
"""
Load Google Ads keyword performance into BigQuery (ads_keyword_performance).

Fields include impressions, clicks, cost, conversions, value, CTR, CPC, keyword text (redacted by default).
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
          campaign.name,
          ad_group.id,
          ad_group.name,
          ad_group_criterion.criterion_id,
          ad_group_criterion.keyword.text,
          ad_group_criterion.keyword.match_type,
          metrics.impressions,
          metrics.clicks,
          metrics.cost_micros,
          metrics.conversions,
          metrics.conversions_value,
          metrics.ctr,
          metrics.average_cpc
        FROM keyword_view
        WHERE segments.date BETWEEN '{start}' AND '{end}'
          AND ad_group_criterion.status != 'REMOVED'
    """


def schema():
    return [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("campaign_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_group_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("ad_group_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("ad_group_name_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("criterion_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("keyword_text", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("keyword_text_hash", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("match_type", "STRING", mode="NULLABLE"),
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
        camp_name = r.campaign.name
        adg_name = r.ad_group.name
        kw_text = r.ad_group_criterion.keyword.text
        rn = {
            "date": r.segments.date,
            "customer_id": r.customer.id,
            "campaign_id": str(r.campaign.id),
            "ad_group_id": str(r.ad_group.id),
            "criterion_id": str(r.ad_group_criterion.criterion_id),
            "match_type": str(r.ad_group_criterion.keyword.match_type),
            "impressions": int(r.metrics.impressions),
            "clicks": int(r.metrics.clicks),
            "cost_micros": int(r.metrics.cost_micros),
            "conversions": float(r.metrics.conversions),
            "conversion_value": float(r.metrics.conversions_value),
            "ctr": float(r.metrics.ctr),
            "avg_cpc_micros": int(r.metrics.average_cpc),
        }
        c = _redact(camp_name, enabled=redact_all)
        rn["campaign_name"] = c["text"]
        rn["campaign_name_hash"] = c["hash"]
        g = _redact(adg_name, enabled=redact_all)
        rn["ad_group_name"] = g["text"]
        rn["ad_group_name_hash"] = g["hash"]
        k = _redact(kw_text, enabled=redact_all)
        rn["keyword_text"] = k["text"]
        rn["keyword_text_hash"] = k["hash"]
        out_rows.append(rn)

    ensure_and_insert(project, dataset, "ads_keyword_performance", schema(), out_rows, partition_field="date")
    logger.info(f"Inserted {len(out_rows)} rows into {dataset}.ads_keyword_performance for customer {customer_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--customer", required=True, help="Customer ID (10 digits)")
    args = p.parse_args()
    datetime.strptime(args.start, "%Y-%m-%d")
    datetime.strptime(args.end, "%Y-%m-%d")
    run(args.start, args.end, customer_id=args.customer)

