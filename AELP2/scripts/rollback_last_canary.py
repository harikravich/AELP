#!/usr/bin/env python3
"""
Rollback the last applied canary budget changes using the changefeed in
`<project>.<dataset>.canary_changes`.

Env:
  GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID
  AELP2_ALLOW_GOOGLE_MUTATIONS=1 (required to actually apply rollback)
"""

import os
import sys
from datetime import datetime, timezone
from typing import List, Dict

from AELP2.core.ingestion.bq_loader import get_bq_client

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


def set_budget_amount(client: GoogleAdsClient, customer_id: str, resource_name: str, amount_micros: int):
    svc = client.get_service("CampaignBudgetService")
    op = client.get_type("CampaignBudgetOperation")
    budget = op.update
    budget.resource_name = resource_name
    budget.amount_micros = amount_micros
    mask = client.get_type("FieldMask")
    mask.paths.append("amount_micros")
    op.update_mask = mask
    resp = svc.mutate_campaign_budgets(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def main():
    project = _required("GOOGLE_CLOUD_PROJECT")
    dataset = _required("BIGQUERY_TRAINING_DATASET")

    allow = os.getenv("AELP2_ALLOW_GOOGLE_MUTATIONS", "0") == "1"
    if not allow:
        print("Set AELP2_ALLOW_GOOGLE_MUTATIONS=1 to apply rollback.")
        sys.exit(2)

    developer_token = _required("GOOGLE_ADS_DEVELOPER_TOKEN")
    client_id = _required("GOOGLE_ADS_CLIENT_ID")
    client_secret = _required("GOOGLE_ADS_CLIENT_SECRET")
    refresh_token = _required("GOOGLE_ADS_REFRESH_TOKEN")
    customer_id = _required("GOOGLE_ADS_CUSTOMER_ID").replace("-", "")

    bq = get_bq_client()
    ds = f"{project}.{dataset}"
    # Fetch last run_id with applied changes
    sql = f"""
      SELECT run_id
      FROM `{ds}.canary_changes`
      WHERE applied = TRUE
      ORDER BY timestamp DESC
      LIMIT 1
    """
    rows = list(bq.query(sql).result())
    if not rows:
        print("No applied canary changes found.")
        return
    run_id = rows[0].run_id
    # Fetch all changes for that run_id
    sql2 = f"""
      SELECT campaign_id, old_budget
      FROM `{ds}.canary_changes`
      WHERE run_id = @rid AND applied = TRUE
    """
    job = bq.query(sql2, job_config=bq.query_job_config.from_api_repr({
        'queryParameters': [{'name':'rid','parameterType':{'type':'STRING'},'parameterValue':{'value':run_id}}]
    }))
    changes = [(str(r.campaign_id), float(r.old_budget)) for r in job.result()]
    if not changes:
        print(f"No applied rows for run_id {run_id}")
        return

    cfg = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "use_proto_plus": True,
    }
    client = GoogleAdsClient.load_from_dict(cfg)
    ga_service = client.get_service("GoogleAdsService")

    # Resolve budget resource names for campaigns
    ids_list = ",".join([cid for cid, _ in changes])
    q = f"SELECT campaign.id, campaign.campaign_budget FROM campaign WHERE campaign.id IN ({ids_list})"
    req = client.get_type("SearchGoogleAdsRequest")
    req.customer_id = customer_id
    req.query = q
    rs = {str(r.campaign.id): str(r.campaign.campaign_budget) for r in ga_service.search(request=req)}

    # Apply rollbacks
    for cid, old_budget in changes:
        res_name = rs.get(cid)
        if not res_name:
            print(f"Campaign {cid}: budget resource not found; skipping")
            continue
        new_micros = int(round(old_budget * 1e6))
        try:
            rn = set_budget_amount(client, customer_id, res_name, new_micros)
            print(f"Rolled back campaign {cid} to {old_budget:.2f}: {rn}")
        except Exception as e:
            print(f"Failed rollback for {cid}: {e}")


if __name__ == "__main__":
    main()

