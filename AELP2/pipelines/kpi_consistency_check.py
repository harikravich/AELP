#!/usr/bin/env python3
"""
KPI Consistency Check: compares KPI-only view vs training episodes daily metrics.

Writes `<project>.<dataset>.kpi_consistency_checks` with diffs for CAC/ROAS.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.kpi_consistency_checks"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('kpi_cac', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('train_cac', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('cac_diff', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('kpi_roas', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('train_roas', 'FLOAT', mode='NULLABLE'),
            bigquery.SchemaField('roas_diff', 'FLOAT', mode='NULLABLE'),
        ]
        t = bigquery.Table(table_id, schema=schema)
        t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field='date')
        bq.create_table(t)
        return table_id


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    table_id = ensure_table(bq, project, dataset)
    sql = f"""
      WITH kpi AS (
        SELECT date, cac AS kpi_cac, roas AS kpi_roas FROM `{project}.{dataset}.ads_kpi_daily`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      ),
      tr AS (
        SELECT date, cac AS train_cac, roas AS train_roas FROM `{project}.{dataset}.training_episodes_daily`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 28 DAY) AND CURRENT_DATE()
      )
      SELECT kpi.date,
             kpi.kpi_cac,
             tr.train_cac,
             (tr.train_cac - kpi.kpi_cac) AS cac_diff,
             kpi.kpi_roas,
             tr.train_roas,
             (tr.train_roas - kpi.kpi_roas) AS roas_diff
      FROM kpi
      LEFT JOIN tr USING(date)
      ORDER BY kpi.date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if rows:
        for r in rows:
            r['date'] = r['date'].isoformat()
        bq.insert_rows_json(table_id, rows)
        print(f"Inserted {len(rows)} rows into {table_id}")
    else:
        print('No KPI consistency rows computed')


if __name__ == '__main__':
    main()

