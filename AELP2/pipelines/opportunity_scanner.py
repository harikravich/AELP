#!/usr/bin/env python3
"""
Opportunity Scanner v1 (Google-first, shadow)

Surfaces headroom and expansion candidates across:
- Search (brand vs non-brand headroom via impression_share and CPA)
- YouTube/Demand Gen/Discovery (placeholder until adapters enriched)
- PMax (placeholder; gated)

Outputs are logged to `platform_skeletons` with rationale/expected CAC/volume (shadow).

Dry-run mode prints candidates to stdout without BigQuery writes.
"""
import os
import json
from datetime import datetime
from typing import List, Dict

from google.cloud import bigquery


def search_headroom_candidates(bq: bigquery.Client, project: str, dataset: str) -> List[Dict]:
    sql = f"""
      WITH agg AS (
        SELECT 
          CASE WHEN LOWER(campaign_name_hash) LIKE '%brand%' THEN 'brand' ELSE 'non_brand' END AS kind,
          SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS cac,
          APPROX_QUANTILES(impression_share, 100)[OFFSET(50)] AS is_p50,
          SUM(conversions) AS conv,
          SUM(cost_micros)/1e6 AS cost
        FROM `{project}.{dataset}.ads_campaign_performance`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
        GROUP BY kind
      )
      SELECT * FROM agg
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    out = []
    for r in rows:
        kind = r['kind']
        is_p50 = float(r.get('is_p50') or 0.0)
        cac = float(r.get('cac') or 0.0)
        if is_p50 < 0.8 and cac > 0:  # simple headroom heuristic
            out.append({
                'platform': 'google_ads',
                'channel': 'search',
                'objective': f'expand_{kind}',
                'expected_cac': cac,
                'impression_share_p50': is_p50,
                'rationale': f"IS p50={is_p50:.2f} < 0.8; CAC {cac:.2f}; expand budget/queries",
            })
    return out


def log_skeletons(bq: bigquery.Client, project: str, dataset: str, candidates: List[Dict]):
    table_id = f"{project}.{dataset}.platform_skeletons"
    rows = []
    for c in candidates:
        rows.append({
            'timestamp': datetime.utcnow().isoformat(),
            'platform': c['platform'],
            'campaign_name': f"AUTO-{c['objective'].upper()}-{datetime.utcnow().strftime('%Y%m%d')}",
            'objective': c['objective'],
            'daily_budget': float(os.getenv('AELP2_OPP_DEFAULT_BUDGET', '50')),
            'status': 'paused',
            'notes': c['rationale'],
            'utm': json.dumps({'utm_source': c['platform'], 'utm_campaign': c['objective']}),
        })
    if rows:
        bq.insert_rows_json(table_id, rows)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)

    candidates = []
    try:
        candidates += search_headroom_candidates(bq, project, dataset)
    except Exception as e:
        print(f"search_headroom_candidates error: {e}")

    if args.dry_run:
        print(json.dumps({'candidates': candidates}, indent=2))
        return
    log_skeletons(bq, project, dataset, candidates)
    print(f"Logged {len(candidates)} opportunity skeletons to {project}.{dataset}.platform_skeletons")


if __name__ == '__main__':
    main()

