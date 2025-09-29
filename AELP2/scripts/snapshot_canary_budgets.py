#!/usr/bin/env python3
"""
Snapshot current budgets for a set of Google Ads campaigns into BigQuery
table `<project>.<dataset>.canary_budgets_snapshot`.

Env:
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID (10 digits)
  AELP2_GOOGLE_CANARY_CAMPAIGN_IDS: comma-separated campaign IDs
"""

import os
import sys
from datetime import datetime, timezone
from typing import List

from AELP2.core.ingestion.bq_loader import get_bq_client, ensure_dataset, ensure_table

try:
    from google.ads.googleads.client import GoogleAdsClient
except Exception as e:
    print(f"google-ads not installed: {e}", file=sys.stderr)
    sys.exit(2)


def _required(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"Missing required env: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def enumerate_budgets(client: GoogleAdsClient, customer_id: str, campaign_ids: List[str]):
    ga_service = client.get_service("GoogleAdsService")
    ids_list = ",".join(campaign_ids)
    query = f"""
        SELECT campaign.id, campaign.name, campaign.campaign_budget, campaign_budget.amount_micros
        FROM campaign
        WHERE campaign.id IN ({ids_list})
    """
    request = client.get_type("SearchGoogleAdsRequest")
    request.customer_id = customer_id
    request.query = query
    return list(ga_service.search(request=request))


def main():
    project = _required("GOOGLE_CLOUD_PROJECT")
    dataset = _required("BIGQUERY_TRAINING_DATASET")
    developer_token = _required("GOOGLE_ADS_DEVELOPER_TOKEN")
    client_id = _required("GOOGLE_ADS_CLIENT_ID")
    client_secret = _required("GOOGLE_ADS_CLIENT_SECRET")
    refresh_token = _required("GOOGLE_ADS_REFRESH_TOKEN")
    customer_id = _required("GOOGLE_ADS_CUSTOMER_ID").replace("-", "")
    ids_csv = _required("AELP2_GOOGLE_CANARY_CAMPAIGN_IDS")
    campaign_ids = [x.strip() for x in ids_csv.split(',') if x.strip()]
    if not campaign_ids:
        print("No campaign IDs provided", file=sys.stderr)
        sys.exit(2)

    cfg = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "use_proto_plus": True,
    }
    client = GoogleAdsClient.load_from_dict(cfg)

    rows = enumerate_budgets(client, customer_id, campaign_ids)
    bq = get_bq_client()
    ds_id = f"{project}.{dataset}"
    ensure_dataset(bq, ds_id)
    from google.cloud import bigquery
    snap_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("budget", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("amount_micros", "INT64", mode="NULLABLE"),
    ]
    ensure_table(bq, f"{ds_id}.canary_budgets_snapshot", snap_schema, partition_field="timestamp")

    ts = datetime.now(timezone.utc).isoformat()
    out = []
    for row in rows:
        out.append({
            "timestamp": ts,
            "customer_id": customer_id,
            "campaign_id": str(row.campaign.id),
            "campaign_name": str(row.campaign.name),
            "budget": float(int(row.campaign_budget.amount_micros)/1e6),
            "amount_micros": int(row.campaign_budget.amount_micros),
        })
    if out:
        bq.insert_rows_json(f"{ds_id}.canary_budgets_snapshot", out)
        print(f"Wrote {len(out)} snapshot rows to {ds_id}.canary_budgets_snapshot")
    else:
        print("No matching campaigns found to snapshot.")


if __name__ == "__main__":
    main()
