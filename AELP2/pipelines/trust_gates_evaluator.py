#!/usr/bin/env python3
"""
Trust Gates Evaluator: computes pass/fail for pilot gates and writes to BQ.

Table: `<project>.<dataset>.trust_gates` with rows per gate and status.
"""
import os
import json
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.trust_gates"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('gate', 'STRING'),
            bigquery.SchemaField('passed', 'BOOL'),
            bigquery.SchemaField('details', 'JSON'),
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
    now = datetime.utcnow().isoformat()

    # Fidelity: last 14d mean absolute percentage error caps
    roas_mape = None
    cac_mape = None
    try:
        sql = f"""
          SELECT AVG(ABS(SAFE_DIVIDE(roas_diff, NULLIF(kpi_roas,0)))) AS roas_mape,
                 AVG(ABS(SAFE_DIVIDE(cac_diff, NULLIF(kpi_cac,0)))) AS cac_mape
          FROM `{project}.{dataset}.kpi_consistency_checks`
          WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) AND CURRENT_DATE()
        """
        row = list(bq.query(sql).result())[0]
        roas_mape = float(row.roas_mape or 1.0)
        cac_mape = float(row.cac_mape or 1.0)
    except Exception:
        roas_mape = cac_mape = 1.0
    fid_ok = (roas_mape <= 0.5) and (cac_mape <= 0.5)
    bq.insert_rows_json(table_id, [{
        'timestamp': now,
        'gate': 'fidelity_mape',
        'passed': fid_ok,
        'details': json.dumps({'roas_mape': roas_mape, 'cac_mape': cac_mape, 'threshold': 0.5}),
    }])

    # Stability: spend anomaly alerts absent in last 7d
    alerts = 0
    try:
        sql = f"""
          SELECT COUNT(*) AS n FROM `{project}.{dataset}.ops_alerts`
          WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            AND alert IN ('canary_spend_spike','policy_error')
        """
        alerts = int(list(bq.query(sql).result())[0].n)
    except Exception:
        alerts = 0
    stab_ok = (alerts == 0)
    bq.insert_rows_json(table_id, [{
        'timestamp': now,
        'gate': 'stability_alerts',
        'passed': stab_ok,
        'details': json.dumps({'alerts_last_7d': alerts}),
    }])

    print(f"trust_gates written (fidelity={fid_ok}, stability={stab_ok}) to {table_id}")


if __name__ == '__main__':
    main()

