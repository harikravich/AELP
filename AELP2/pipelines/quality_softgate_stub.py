#!/usr/bin/env python3
"""
Quality soft-gate (stub): reads quality_signal_daily and emits safety_events when below threshold.
Env: AELP2_QUALITY_MIN (float 0..1)
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
    thr = float(os.getenv('AELP2_QUALITY_MIN', '0.0') or 0)
    if thr <= 0:
        print('No threshold set; skipping')
        return
    bq = bigquery.Client(project=project)
    sql = f"""
      SELECT AVG(score) AS avg_score
      FROM `{project}.{dataset}.quality_signal_daily`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE()
    """
    try:
        avg = 1.0
        for r in bq.query(sql).result():
            avg = float(r.avg_score or 1.0)
    except Exception:
        avg = 1.0
    if avg < thr:
        table = f"{project}.{dataset}.safety_events"
        try:
            bq.get_table(table)
        except Exception:
            from google.cloud import bigquery as bq2
            t = bq2.Table(table, schema=[
                bq2.SchemaField('timestamp', 'TIMESTAMP'),
                bq2.SchemaField('event_type', 'STRING'),
                bq2.SchemaField('severity', 'STRING'),
                bq2.SchemaField('episode_id', 'STRING'),
                bq2.SchemaField('metadata', 'JSON'),
                bq2.SchemaField('action_taken', 'STRING'),
            ])
            t.time_partitioning = bq2.TimePartitioning(type_=bq2.TimePartitioningType.DAY, field='timestamp')
            bq.create_table(t)
        bq.insert_rows_json(table, [{
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'quality_soft_gate_breach',
            'severity': 'WARNING',
            'episode_id': None,
            'metadata': f'{{"avg": {avg}, "min": {thr}}}',
            'action_taken': 'alert_only',
        }])
        print('Quality soft-gate alert written')
    else:
        print('Quality within threshold')


if __name__ == '__main__':
    main()

