#!/usr/bin/env python3
"""
Parity Report (stub): compares ads_kpi_daily vs training_episodes_daily.

Writes `<project>.<dataset>.parity_reports` with daily RMSE and rel diffs.
"""
import os, math
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.parity_reports"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('rmse_cac', 'FLOAT'),
            bigquery.SchemaField('rmse_roas', 'FLOAT'),
            bigquery.SchemaField('rel_diff_cac', 'FLOAT'),
            bigquery.SchemaField('rel_diff_roas', 'FLOAT'),
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
      WITH a AS (
        SELECT date, cac AS cac_a, roas AS roas_a FROM `{project}.{dataset}.ads_kpi_daily`
      ), b AS (
        SELECT date, cac AS cac_b, roas AS roas_b FROM `{project}.{dataset}.training_episodes_daily`
      ), j AS (
        SELECT a.date, a.cac_a, b.cac_b, a.roas_a, b.roas_b
        FROM a JOIN b USING(date)
        WHERE a.date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      )
      SELECT date,
             POW(POW(IFNULL(cac_b,0)-IFNULL(cac_a,0),2),0.5) AS rmse_cac,
             POW(POW(IFNULL(roas_b,0)-IFNULL(roas_a,0),2),0.5) AS rmse_roas,
             SAFE_DIVIDE(IFNULL(cac_b,0)-IFNULL(cac_a,0), NULLIF(cac_a,0)) AS rel_diff_cac,
             SAFE_DIVIDE(IFNULL(roas_b,0)-IFNULL(roas_a,0), NULLIF(roas_a,0)) AS rel_diff_roas
      FROM j
      ORDER BY date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    for r in rows:
        r['date'] = r['date'].isoformat()
    if rows:
        bq.insert_rows_json(table_id, rows)
        print(f'Wrote {len(rows)} parity rows to {table_id}')
    else:
        print('No parity rows computed')


if __name__ == '__main__':
    main()

