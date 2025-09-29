#!/usr/bin/env python3
"""
Copy Optimization Loop (policy-safe, stub).

Generates simple copy suggestions based on recent CTR deltas by campaign.
Writes `<project>.<dataset>.copy_suggestions` with shadow-only proposals.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.copy_suggestions"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('suggestion', 'STRING'),
            bigquery.SchemaField('rationale', 'STRING'),
            bigquery.SchemaField('policy_safe', 'BOOL'),
            bigquery.SchemaField('shadow', 'BOOL'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    try:
        sql = f"""
          WITH d AS (
            SELECT CAST(campaign_id AS STRING) AS cid,
                   SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr
            FROM `{project}.{dataset}.ads_campaign_performance`
            WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
            GROUP BY cid
          ),
          w AS (
            SELECT CAST(campaign_id AS STRING) AS cid,
                   SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr
            FROM `{project}.{dataset}.ads_campaign_performance`
            WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 15 DAY)
            GROUP BY cid
          )
          SELECT d.cid, d.ctr AS ctr_recent, w.ctr AS ctr_prev,
                 SAFE_DIVIDE(d.ctr - w.ctr, NULLIF(w.ctr,0)) AS delta
          FROM d LEFT JOIN w USING(cid)
          WHERE SAFE_DIVIDE(d.ctr - w.ctr, NULLIF(w.ctr,0)) < -0.1
          ORDER BY delta ASC
          LIMIT 10
        """
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows = []
    now = datetime.utcnow().isoformat()
    out = []
    if rows:
        for r in rows:
            out.append({
                'timestamp': now,
                'campaign_id': r['cid'],
                'suggestion': 'Refresh headline with benefit-led copy; test variant',
                'rationale': f"CTR decline {float(r.get('delta') or 0.0):.2f}; propose A/B test",
                'policy_safe': True,
                'shadow': True,
            })
    else:
        out.append({
            'timestamp': now,
            'campaign_id': 'none',
            'suggestion': 'No decline detected; keep monitoring',
            'rationale': 'Stub path (no data) or stable CTR',
            'policy_safe': True,
            'shadow': True,
        })
    bq.insert_rows_json(table_id, out)
    print(f'Wrote {len(out)} copy suggestions to {table_id}')


if __name__ == '__main__':
    main()

