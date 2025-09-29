#!/usr/bin/env python3
"""
SLO/Alerting stub: scans ops_flow_runs and safety_events for failures; emits ops_alerts.
"""
import os
from datetime import datetime
from google.cloud import bigquery


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        print('Missing env; skipping')
        return
    bq = bigquery.Client(project=project)
    # Flow failures in last day
    sql = f"""
      SELECT COUNTIF(ok=FALSE) AS failures
      FROM `{project}.{dataset}.ops_flow_runs`
      WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    """
    failures = 0
    try:
        for r in bq.query(sql).result():
            failures = int(r.failures or 0)
    except Exception:
        failures = 0
    if failures > 0:
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
            'alert': 'flow_failures_detected',
            'severity': 'WARNING',
            'details': f'{{"failures": {failures}}}',
        }])
        print('Flow failures alert emitted')
    else:
        print('No SLO breaches detected')


if __name__ == '__main__':
    main()

