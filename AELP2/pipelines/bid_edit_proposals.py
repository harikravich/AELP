#!/usr/bin/env python3
"""
Bid Edit Proposals (shadow-only, stub).

Proposes ad-group max CPC adjustments based on simple CPC vs CTR proxy.
Writes `<project>.<dataset>.bid_edit_proposals` with shadow=true.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.bid_edit_proposals"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('ad_group_id', 'STRING'),
            bigquery.SchemaField('current_cpc', 'FLOAT'),
            bigquery.SchemaField('proposed_cpc', 'FLOAT'),
            bigquery.SchemaField('delta_pct', 'FLOAT'),
            bigquery.SchemaField('rationale', 'STRING'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        bq.create_table(bigquery.Table(table_id, schema=schema))
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS campaign_id,
             CAST(ad_group_id AS STRING) AS ad_group_id,
             SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(clicks),0)) AS cpc,
             SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr
      FROM `{project}.{dataset}.ads_ad_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
      GROUP BY campaign_id, ad_group_id
      HAVING ctr < 0.02 AND cpc > 1.00
      ORDER BY cpc DESC
      LIMIT 10
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    out = []
    now = datetime.utcnow().isoformat()
    for r in rows:
        current_cpc = float(r.get('cpc') or 0.0)
        proposed = max(0.1, current_cpc * 0.9)
        out.append({
            'timestamp': now,
            'campaign_id': r['campaign_id'],
            'ad_group_id': r['ad_group_id'],
            'current_cpc': current_cpc,
            'proposed_cpc': proposed,
            'delta_pct': (proposed - current_cpc) / max(current_cpc, 1e-9),
            'rationale': 'CTR<2% and CPC>1.0; propose -10% CPC (shadow-only)',
            'shadow': True,
        })
    if out:
        bq.insert_rows_json(table_id, out)
        print(f'Wrote {len(out)} bid edit proposals to {table_id}')
    else:
        print('No bid edit proposals (criteria not met)')


if __name__ == '__main__':
    main()

