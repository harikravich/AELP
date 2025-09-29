#!/usr/bin/env python3
"""
Cross-Platform KPI Daily (stub): aggregates KPI metrics across platforms.

For now, includes only Google Ads data into `<project>.<dataset>.cross_platform_kpi_daily`.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.cross_platform_kpi_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('date', 'DATE'),
            bigquery.SchemaField('platform', 'STRING'),
            bigquery.SchemaField('cost', 'FLOAT'),
            bigquery.SchemaField('conversions', 'FLOAT'),
            bigquery.SchemaField('revenue', 'FLOAT'),
            bigquery.SchemaField('cac', 'FLOAT'),
            bigquery.SchemaField('roas', 'FLOAT'),
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
      SELECT date, 'google_ads' AS platform, cost, conversions, revenue, cac, roas
      FROM `{project}.{dataset}.ads_kpi_daily`
      WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ORDER BY date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if rows:
        for r in rows:
            r['date'] = r['date'].isoformat()
        bq.insert_rows_json(table_id, rows)
        print(f'Wrote {len(rows)} cross-platform KPI rows to {table_id}')
    else:
        print('No KPI rows to write')


if __name__ == '__main__':
    main()

