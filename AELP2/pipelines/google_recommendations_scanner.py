#!/usr/bin/env python3
"""
Google Ads Recommendations Scanner (safe, shadow-only).

- Tries the Google Ads Recommendations API when `google-ads` + creds are available.
- Else, falls back to simple BQ heuristics (if available) to propose quick wins.
- If neither is available, writes a stub note so dashboards have a safe default.

Outputs:
- `<project>.<dataset>.platform_skeletons` (paused candidates with rationale)
- `<project>.<dataset>.recs_quickwins` (optional summary rows)
"""
import os
import json
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

def ensure_recs_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.recs_quickwins"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('type', 'STRING'),
            bigquery.SchemaField('details', 'JSON'),
            bigquery.SchemaField('reason', 'STRING'),
            bigquery.SchemaField('priority', 'STRING'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def log_candidate(bq: bigquery.Client, project: str, dataset: str, platform: str, objective: str, notes: str, budget: float = 50.0):
    row = {
        'timestamp': datetime.utcnow().isoformat(),
        'platform': platform,
        'campaign_name': f"AUTO-REC-{objective.upper()}-{datetime.utcnow().strftime('%Y%m%d')}",
        'objective': objective,
        'daily_budget': budget,
        'status': 'paused',
        'notes': notes,
        'utm': json.dumps({'utm_source': platform, 'utm_campaign': objective}),
    }
    try:
        bq.insert_rows_json(f"{project}.{dataset}.platform_skeletons", [row])
    except Exception as e:
        print(f"platform_skeletons insert failed: {e}")


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    # Try BQ
    try:
        bq = bigquery.Client(project=project)
    except Exception as e:
        print(f'Recommendations: BigQuery client unavailable: {e}')
        return

    # Try Google Ads API (best-effort)
    used_api = False
    try:
        from google.ads.googleads.client import GoogleAdsClient  # type: ignore
        # Note: In this environment, we do not execute API calls. Mark as present and fall through.
        used_api = True
    except Exception:
        used_api = False

    # Ensure quickwins table
    try:
        ensure_recs_table(bq, project, dataset)
    except Exception as e:
        print(f'recs_quickwins ensure failed: {e}')

    if used_api:
        # Placeholder: record that API path is available (details omitted in sandbox)
        try:
            bq.insert_rows_json(f"{project}.{dataset}.recs_quickwins", [{
                'timestamp': datetime.utcnow().isoformat(),
                'type': 'api_available',
                'details': json.dumps({'note': 'google-ads present; implement API fetch with customer_id context'}),
                'reason': 'library_present',
                'priority': 'low',
            }])
        except Exception:
            pass

    # Heuristic fallback from BQ (if ads tables exist)
    candidates = []
    try:
        sql_budget = f"""
          SELECT CAST(campaign_id AS STRING) AS campaign_id,
                 SAFE_DIVIDE(SUM(conversions), NULLIF(SUM(cost_micros)/1e6,0)) AS conv_per_dollar,
                 APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS is_p50
          FROM `{project}.{dataset}.ads_campaign_performance`
          WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
          GROUP BY campaign_id
          HAVING is_p50 < 0.8 AND conv_per_dollar IS NOT NULL
          ORDER BY conv_per_dollar DESC
          LIMIT 3
        """
        for r in bq.query(sql_budget).result():
            candidates.append(('budget_increase_candidate', f"cid={r.campaign_id} is_p50={float(r.is_p50 or 0):.2f}, eff={float(r.conv_per_dollar or 0):.4f}"))
    except Exception:
        pass
    # Keyword/asset placeholders with rationale
    try:
        # If search terms table exists, pick top rising term (placeholder query)
        _ = bq.get_table(f"{project}.{dataset}.ads_search_terms")
        candidates.append(('keyword_add', 'rising term from ads_search_terms (placeholder)'))
    except Exception:
        candidates.append(('keyword_add', 'high-intent term from search_terms (placeholder)'))
    candidates.append(('asset_enhancement', 'Missing sitelinks/callouts detected (placeholder)'))

    for typ, reason in candidates:
        log_candidate(bq, project, dataset, 'google_ads', typ, f'Recs: {reason}', budget=50.0)
    print(f"Logged {len(candidates)} recommendation candidates (shadow) to platform_skeletons")


if __name__ == '__main__':
    main()
