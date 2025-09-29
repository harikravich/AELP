#!/usr/bin/env python3
"""
Bid Landscape Modeling (stub): derive CPC↔volume curves per campaign.

Writes `<project>.<dataset>.bid_landscape_curves` with simple log curves.
"""
import os
import math
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.bid_landscape_curves"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('cpc', 'FLOAT'),
            bigquery.SchemaField('expected_clicks', 'FLOAT'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    import datetime
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    sql = f"""
      SELECT CAST(campaign_id AS STRING) AS campaign_id,
             SAFE_DIVIDE(SUM(clicks), NULLIF(SUM(impressions),0)) AS ctr,
             SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(clicks),0)) AS cpc
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      GROUP BY campaign_id
      ORDER BY campaign_id
      LIMIT 20
    """
    try:
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows = []
    now = datetime.datetime.utcnow().isoformat()
    out = []
    for r in rows:
        base_cpc = float(r.get('cpc') or 1.0)
        base_ctr = float(r.get('ctr') or 0.02)
        for m in [0.5, 0.75, 1.0, 1.25, 1.5]:
            cpc = max(0.1, base_cpc * m)
            expected_clicks = 1000.0 * base_ctr * math.log(1.0 + m * 2.0)
            out.append({'timestamp': now, 'campaign_id': r['campaign_id'], 'cpc': cpc, 'expected_clicks': expected_clicks})
    if out:
        bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} curve points to {table_id}")


if __name__ == '__main__':
    main()

