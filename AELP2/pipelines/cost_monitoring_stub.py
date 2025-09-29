#!/usr/bin/env python3
"""
Cost monitoring (stub): compute daily cost and emit ops_alerts if exceeding cap.
Env: AELP2_DAILY_COST_CAP (float)
"""
import os
from datetime import datetime
from google.cloud import bigquery


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    cap = float(os.getenv('AELP2_DAILY_COST_CAP', '0') or 0)
    if not project or not dataset or cap <= 0:
        print('No project/dataset or cap; skipping')
        return
    bq = bigquery.Client(project=project)
    sql = f"""
      SELECT SUM(cost_micros)/1e6 AS cost
      FROM `{project}.{dataset}.ads_campaign_performance`
      WHERE DATE(date) = CURRENT_DATE()
    """
    cost = 0.0
    for r in bq.query(sql).result():
        cost = float(r.cost or 0.0)
    if cost > cap:
        table = f"{project}.{dataset}.ops_alerts"
        try:
            bq.get_table(table)
        except Exception:
            from google.cloud import bigquery as bq2
            t = bq2.Table(table, schema=[
                bq2.SchemaField('timestamp', 'TIMESTAMP'),
                bq2.SchemaField('alert', 'STRING'),
                bq2.SchemaField('severity', 'STRING'),
                bq2.SchemaField('details', 'JSON'),
            ])
            t.time_partitioning = bq2.TimePartitioning(type_=bq2.TimePartitioningType.DAY, field='timestamp')
            bq.create_table(t)
        bq.insert_rows_json(table, [{
            'timestamp': datetime.utcnow().isoformat(),
            'alert': 'daily_cost_cap_exceeded',
            'severity': 'WARNING',
            'details': f'{{"cost": {cost}, "cap": {cap}}}',
        }])
        print('Cost cap exceeded alert emitted')
    else:
        print('Cost within cap')


if __name__ == '__main__':
    main()

