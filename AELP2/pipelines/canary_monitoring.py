#!/usr/bin/env python3
"""
Canary Monitoring (shadow): detect anomalies and write ops_alerts.

Rules (simple):
- Spend spike on canary campaigns vs prior 7d avg > AELP2_ALERT_SPEND_DELTA_PCT (default 0.5)

Writes `<project>.<dataset>.ops_alerts`.
"""
import os
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.ops_alerts"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('alert', 'STRING'),
            bigquery.SchemaField('severity', 'STRING'),
            bigquery.SchemaField('details', 'JSON'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='timestamp')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    canary_ids = os.getenv('AELP2_GOOGLE_CANARY_CAMPAIGN_IDS', '')
    if not project or not dataset or not canary_ids:
        print('Missing env or no canary IDs; skipping')
        return
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    ids = ",".join([f"'{x.strip()}'" for x in canary_ids.split(',') if x.strip()])
    thresh = float(os.getenv('AELP2_ALERT_SPEND_DELTA_PCT', '0.5'))
    sql = f"""
      WITH cur AS (
        SELECT CAST(campaign_id AS STRING) AS cid, SUM(cost_micros)/1e6 AS cost
        FROM `{project}.{dataset}.ads_campaign_performance`
        WHERE DATE(date) = CURRENT_DATE() AND CAST(campaign_id AS STRING) IN ({ids})
        GROUP BY cid
      ),
      hist AS (
        SELECT CAST(campaign_id AS STRING) AS cid, AVG(cost_micros/1e6) AS avg_cost
        FROM `{project}.{dataset}.ads_campaign_performance`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
          AND CAST(campaign_id AS STRING) IN ({ids})
        GROUP BY cid
      )
      SELECT cur.cid, cur.cost, hist.avg_cost,
             SAFE_DIVIDE(cur.cost - hist.avg_cost, NULLIF(hist.avg_cost,0)) AS delta
      FROM cur LEFT JOIN hist USING(cid)
      WHERE SAFE_DIVIDE(cur.cost - hist.avg_cost, NULLIF(hist.avg_cost,0)) > {thresh}
    """
    alerts = [dict(r) for r in bq.query(sql).result()]
    for a in alerts:
        row = {
            'timestamp': datetime.utcnow().isoformat(),
            'alert': 'canary_spend_spike',
            'severity': 'WARNING',
            'details': str({'cid': a['cid'], 'cost': float(a.get('cost') or 0.0), 'avg_cost': float(a.get('avg_cost') or 0.0), 'delta': float(a.get('delta') or 0.0)}),
        }
        bq.insert_rows_json(table_id, [row])
    print(f'Canary monitoring alerts inserted: {len(alerts)}')


if __name__ == '__main__':
    main()

