#!/usr/bin/env python3
"""
Apply safe, canary budget changes to Google Ads campaigns.

Defaults to shadow mode (dry run). Set AELP2_ALLOW_GOOGLE_MUTATIONS=1 and
AELP2_SHADOW_MODE=0 to perform real mutations. Guardrails enforce a maximum
delta percent and a max number of changes per run.

Env:
  GOOGLE_ADS_DEVELOPER_TOKEN, GOOGLE_ADS_CLIENT_ID, GOOGLE_ADS_CLIENT_SECRET,
  GOOGLE_ADS_REFRESH_TOKEN, GOOGLE_ADS_CUSTOMER_ID (10 digits)
  AELP2_GOOGLE_CANARY_CAMPAIGN_IDS: comma-separated campaign IDs (required)
  AELP2_CANARY_BUDGET_DELTA_PCT: max absolute delta fraction (default 0.10)
  AELP2_CANARY_MAX_CHANGES_PER_RUN: max campaigns to change (default 5)
  AELP2_ALLOW_GOOGLE_MUTATIONS: '1' to allow real changes (default '0')
  AELP2_SHADOW_MODE: '1' to force shadow (default '1')
"""

import os
import sys
import uuid
from typing import List, Dict
from datetime import datetime, timezone

from AELP2.core.ingestion.bq_loader import get_bq_client, ensure_dataset, ensure_table
from AELP2.core.safety.feature_gates import is_action_allowed, gate_reason

# Optional import so we can support DRY_RUN without requiring the SDK
try:
    from google.ads.googleads.client import GoogleAdsClient  # type: ignore
    _GOOGLE_ADS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    print(f"google-ads not installed (ok for DRY_RUN): {e}", file=sys.stderr)
    _GOOGLE_ADS_AVAILABLE = False


def _required_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"Missing required env: {name}", file=sys.stderr)
        sys.exit(2)
    return v


def get_campaign_budgets(client: GoogleAdsClient, customer_id: str, campaign_ids: List[str]):
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


def set_budget_amount(client: GoogleAdsClient, customer_id: str, campaign_budget_resource: str, new_amount_micros: int):
    campaign_budget_service = client.get_service("CampaignBudgetService")
    op = client.get_type("CampaignBudgetOperation")
    budget = op.update
    budget.resource_name = campaign_budget_resource
    budget.amount_micros = new_amount_micros
    mask = client.get_type("FieldMask")
    mask.paths.append("amount_micros")
    op.update_mask = mask
    resp = campaign_budget_service.mutate_campaign_budgets(customer_id=customer_id, operations=[op])
    return resp.results[0].resource_name


def _ensure_canary_tables(project: str, dataset: str):
    from google.cloud import bigquery
    client = get_bq_client()
    ds_id = f"{project}.{dataset}"
    ensure_dataset(client, ds_id)
    changes_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("old_budget", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("new_budget", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("delta_pct", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("direction", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("shadow", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("applied", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("error", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("actor", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("notes", "STRING", mode="NULLABLE"),
    ]
    ensure_table(client, f"{ds_id}.canary_changes", changes_schema, partition_field="timestamp")
    snap_schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("customer_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("campaign_name", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("budget", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("amount_micros", "INT64", mode="NULLABLE"),
    ]
    ensure_table(client, f"{ds_id}.canary_budgets_snapshot", snap_schema, partition_field="timestamp")


def _write_change_row(project: str, dataset: str, row: Dict):
    client = get_bq_client()
    table_id = f"{project}.{dataset}.canary_changes"
    client.insert_rows_json(table_id, [row])


def _sum_today_delta_pct(project: str, dataset: str, campaign_id: str) -> float:
    client = get_bq_client()
    sql = f"""
    SELECT SAFE_SUM(delta_pct) AS s
    FROM `{project}.{dataset}.canary_changes`
    WHERE campaign_id = @cid AND DATE(timestamp) = CURRENT_DATE()
      AND applied = TRUE
    """
    job = client.query(sql, job_config=client.query_job_config.from_api_repr({
        'queryParameters': [{'name':'cid','parameterType':{'type':'STRING'},'parameterValue':{'value':campaign_id}}]
    }))
    for r in job.result():
        return float(r.s or 0.0)
    return 0.0


def main():
    # Required envs
    developer_token = _required_env("GOOGLE_ADS_DEVELOPER_TOKEN")
    client_id = _required_env("GOOGLE_ADS_CLIENT_ID")
    client_secret = _required_env("GOOGLE_ADS_CLIENT_SECRET")
    refresh_token = _required_env("GOOGLE_ADS_REFRESH_TOKEN")
    customer_id = _required_env("GOOGLE_ADS_CUSTOMER_ID").replace("-", "")

    canary_ids_csv = _required_env("AELP2_GOOGLE_CANARY_CAMPAIGN_IDS")
    canary_ids = [x.strip() for x in canary_ids_csv.split(',') if x.strip()]
    if not canary_ids:
        print("AELP2_GOOGLE_CANARY_CAMPAIGN_IDS has no IDs", file=sys.stderr)
        sys.exit(2)

    try:
        # Respect global canary caps default (≤5% per change)
        max_delta = float(os.getenv("AELP2_CANARY_BUDGET_DELTA_PCT", "0.05"))
    except Exception:
        max_delta = 0.10
    try:
        # Default to one campaign/change per run
        max_changes = int(os.getenv("AELP2_CANARY_MAX_CHANGES_PER_RUN", "1"))
    except Exception:
        max_changes = 5

    allow_mutations = os.getenv("AELP2_ALLOW_GOOGLE_MUTATIONS", "0") == "1"
    if allow_mutations and not is_action_allowed('apply_google_budget'):
        print(f"Gate denied for apply_google_budget: {gate_reason('apply_google_budget')}. Forcing shadow mode.")
        allow_mutations = False
    shadow_mode = os.getenv("AELP2_SHADOW_MODE", "1") == "1"
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or ""
    dataset = os.getenv("BIGQUERY_TRAINING_DATASET") or ""
    actor = os.getenv("AELP2_ACTOR", "cli")
    if project and dataset:
        _ensure_canary_tables(project, dataset)

    cfg = {
        "developer_token": developer_token,
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "use_proto_plus": True,
    }
    login_cid = os.getenv("GOOGLE_ADS_LOGIN_CUSTOMER_ID")
    if login_cid:
        cfg["login_customer_id"] = login_cid.replace("-", "")
    # DRY_RUN mode: skip Ads client calls, just validate gating + BQ ensures
    if os.getenv("DRY_RUN", "0") == "1":
        print("[DRY_RUN] Skipping Google Ads mutations; validated table ensures and gating.")
        # Best-effort ops logging (shadow)
        try:
            if project and dataset:
                from google.cloud import bigquery  # type: ignore
                bq = bigquery.Client(project=project)
                table_id = f"{project}.{dataset}.ops_flow_runs"
                try:
                    bq.get_table(table_id)
                except Exception:
                    schema = [
                        bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                        bigquery.SchemaField('flow', 'STRING'),
                        bigquery.SchemaField('rc_map', 'JSON'),
                        bigquery.SchemaField('failures', 'JSON'),
                        bigquery.SchemaField('ok', 'BOOL'),
                    ]
                    t = bigquery.Table(table_id, schema=schema)
                    t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                    bq.create_table(t)
                bq.insert_rows_json(table_id, [{
                    'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                    'flow': 'apply_canary_cli',
                    'rc_map': '{"dry_run":0}',
                    'failures': '[]',
                    'ok': True,
                }])
        except Exception:
            pass
        return

    if not _GOOGLE_ADS_AVAILABLE:
        print("google-ads SDK unavailable and not in DRY_RUN; aborting.", file=sys.stderr)
        sys.exit(2)

    client = GoogleAdsClient.load_from_dict(cfg)

    rows = get_campaign_budgets(client, customer_id, canary_ids)
    if not rows:
        print("No canary campaigns found; aborting.")
        return
    print(f"Found {len(rows)} canary campaigns. Max changes this run: {max_changes}")
    run_id = str(uuid.uuid4())

    changes_made = 0
    for row in rows:
        if changes_made >= max_changes:
            break
        camp = row.campaign
        budget_res = str(camp.campaign_budget)
        current_micros = int(row.campaign_budget.amount_micros)
        current = current_micros / 1e6
        # Simple policy: increase or decrease by max_delta based on env flag, default − to be conservative
        direction = os.getenv("AELP2_CANARY_BUDGET_DIRECTION", "down")
        factor = 1.0 - max_delta if direction == "down" else 1.0 + max_delta
        new_amt = max(1.0, current * factor)
        new_micros = int(round(new_amt * 1e6))

        delta_pct = (new_amt - current) / max(current, 1e-9)
        print(f"Campaign {camp.id} '{camp.name}': {current:.2f} -> {new_amt:.2f} (delta {delta_pct*100:.1f}%)")

        # Enforce daily cumulative delta cap per campaign
        if project and dataset:
            try:
                today_sum = _sum_today_delta_pct(project, dataset, str(camp.id))
                max_daily = float(os.getenv("AELP2_CANARY_MAX_DAILY_DELTA_PCT", "0.10"))
                if abs(today_sum + delta_pct) > max_daily:
                    print(f"  Skipping: daily delta cap would be exceeded (today={today_sum:.3f}, this={delta_pct:.3f}, cap={max_daily:.3f})")
                    # Log skipped proposal
                    _write_change_row(project, dataset, {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'run_id': run_id,
                        'customer_id': customer_id,
                        'campaign_id': str(camp.id),
                        'campaign_name': str(camp.name),
                        'old_budget': current,
                        'new_budget': new_amt,
                        'delta_pct': delta_pct,
                        'direction': direction,
                        'shadow': True,
                        'applied': False,
                        'error': 'daily_cap_exceeded',
                        'actor': actor,
                        'notes': None,
                    })
                    continue
            except Exception as _:
                pass

        # Log proposal (shadow or applied)
        if project and dataset:
            _write_change_row(project, dataset, {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'run_id': run_id,
                'customer_id': customer_id,
                'campaign_id': str(camp.id),
                'campaign_name': str(camp.name),
                'old_budget': current,
                'new_budget': new_amt,
                'delta_pct': delta_pct,
                'direction': direction,
                'shadow': (not allow_mutations) or shadow_mode,
                'applied': False,
                'error': None,
                'actor': actor,
                'notes': os.getenv('AELP2_PROPOSAL_NOTES', None),
            })

        if not allow_mutations or shadow_mode:
            continue
        try:
            res = set_budget_amount(client, customer_id, budget_res, new_micros)
            print(f"  Updated budget: {res}")
            changes_made += 1
            if project and dataset:
                _write_change_row(project, dataset, {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'run_id': run_id,
                    'customer_id': customer_id,
                    'campaign_id': str(camp.id),
                    'campaign_name': str(camp.name),
                    'old_budget': current,
                    'new_budget': new_amt,
                    'delta_pct': delta_pct,
                    'direction': direction,
                    'shadow': False,
                    'applied': True,
                    'error': None,
                    'actor': actor,
                    'notes': res,
                })
        except Exception as e:
            print(f"  Failed to update budget: {e}", file=sys.stderr)
            if project and dataset:
                _write_change_row(project, dataset, {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'run_id': run_id,
                    'customer_id': customer_id,
                    'campaign_id': str(camp.id),
                    'campaign_name': str(camp.name),
                    'old_budget': current,
                    'new_budget': new_amt,
                    'delta_pct': delta_pct,
                    'direction': direction,
                    'shadow': False,
                    'applied': False,
                    'error': str(e),
                    'actor': actor,
                    'notes': None,
                })

    print(f"Done. Changes applied: {changes_made} (allow_mutations={allow_mutations}, shadow_mode={shadow_mode}) at {datetime.utcnow().isoformat()}Z")
    # Best-effort ops logging (shadow-friendly)
    try:
        if project and dataset:
            from google.cloud import bigquery  # type: ignore
            bq = bigquery.Client(project=project)
            table_id = f"{project}.{dataset}.ops_flow_runs"
            try:
                bq.get_table(table_id)
            except Exception:
                schema = [
                    bigquery.SchemaField('timestamp', 'TIMESTAMP'),
                    bigquery.SchemaField('flow', 'STRING'),
                    bigquery.SchemaField('rc_map', 'JSON'),
                    bigquery.SchemaField('failures', 'JSON'),
                    bigquery.SchemaField('ok', 'BOOL'),
                ]
                t = bigquery.Table(table_id, schema=schema)
                t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
                bq.create_table(t)
            rc_map = {"changes_applied": changes_made}
            bq.insert_rows_json(table_id, [{
                'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
                'flow': 'apply_canary_cli',
                'rc_map': __import__('json').dumps(rc_map),
                'failures': '[]',
                'ok': True,
            }])
    except Exception:
        pass


if __name__ == "__main__":
    main()
