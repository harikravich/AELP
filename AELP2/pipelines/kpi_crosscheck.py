#!/usr/bin/env python3
"""
KPI Cross-check: compare ads_kpi_daily view vs aggregated ads_campaign_performance.

Writes results to `<project>.<dataset>.kpi_crosscheck_daily` with per-day diffs and status.
"""
import os
from google.cloud import bigquery


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.kpi_crosscheck_daily"
    try:
        bq.get_table(table_id)
        return table_id
    except Exception:
        schema = [
            bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
            bigquery.SchemaField('kpi_cac', 'FLOAT'),
            bigquery.SchemaField('agg_cac', 'FLOAT'),
            bigquery.SchemaField('diff_cac', 'FLOAT'),
            bigquery.SchemaField('kpi_roas', 'FLOAT'),
            bigquery.SchemaField('agg_roas', 'FLOAT'),
            bigquery.SchemaField('diff_roas', 'FLOAT'),
            bigquery.SchemaField('status', 'STRING'),
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
    base = f"{project}.{dataset}"
    sql = f"""
      WITH kpi AS (
        SELECT date, cac AS kpi_cac, roas AS kpi_roas FROM `{base}.ads_kpi_daily`
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
      ), agg AS (
        SELECT DATE(date) AS date,
               SAFE_DIVIDE(SUM(cost_micros)/1e6, NULLIF(SUM(conversions),0)) AS agg_cac,
               SAFE_DIVIDE(SUM(conversion_value), NULLIF(SUM(cost_micros)/1e6,0)) AS agg_roas
        FROM `{base}.ads_campaign_performance`
        WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
        GROUP BY date
      )
      SELECT a.date,
             kpi.kpi_cac, a.agg_cac, (a.agg_cac - kpi.kpi_cac) AS diff_cac,
             kpi.kpi_roas, a.agg_roas, (a.agg_roas - kpi.kpi_roas) AS diff_roas,
             CASE WHEN ABS(SAFE_DIVIDE(a.agg_cac - kpi.kpi_cac, NULLIF(kpi.kpi_cac,0))) < 0.15
                    AND ABS(SAFE_DIVIDE(a.agg_roas - kpi.kpi_roas, NULLIF(kpi.kpi_roas,0))) < 0.15
                  THEN 'OK' ELSE 'DRIFT' END AS status
      FROM agg a LEFT JOIN kpi kpi USING(date)
      ORDER BY a.date
    """
    rows = [dict(r) for r in bq.query(sql).result()]
    if rows:
        bq.insert_rows_json(table_id, rows)
        print(f"Wrote {len(rows)} kpi crosscheck rows to {table_id}")
    else:
        print('No KPI rows to cross-check')


if __name__ == '__main__':
    main()

