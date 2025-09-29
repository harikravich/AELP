#!/usr/bin/env python3
"""
Opportunity Outcomes (stub): summarize approvals and subsequent performance.

Writes `<project>.<dataset>.opportunity_outcomes` with counts and recent spend after approvals.
If inputs missing, ensures table and exits.
"""
import os
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str) -> str:
    table_id = f"{project}.{dataset}.opportunity_outcomes"
    try:
        bq.get_table(table_id)
        return table_id
    except NotFound:
        schema = [
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('objective', 'STRING'),
            bigquery.SchemaField('approvals', 'INT64'),
            bigquery.SchemaField('denials', 'INT64'),
            bigquery.SchemaField('recent_cost', 'FLOAT'),
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

    try:
        sql = f"""
          WITH appr AS (
            SELECT objective,
                   SUM(CASE WHEN action='approve' THEN 1 ELSE 0 END) AS approvals,
                   SUM(CASE WHEN action='deny' THEN 1 ELSE 0 END) AS denials
            FROM `{project}.{dataset}.opportunity_approvals`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
            GROUP BY objective
          ),
          spend AS (
            SELECT 'all' AS objective,
                   SUM(cost_micros)/1e6 AS recent_cost
            FROM `{project}.{dataset}.ads_campaign_performance`
            WHERE DATE(date) BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY) AND CURRENT_DATE()
          )
          SELECT CURRENT_TIMESTAMP() AS timestamp,
                 a.objective,
                 a.approvals,
                 a.denials,
                 s.recent_cost
          FROM appr a
          LEFT JOIN spend s ON TRUE
        """
        rows = [dict(r) for r in bq.query(sql).result()]
    except Exception:
        rows = []
    if rows:
        for r in rows:
            bq.insert_rows_json(table_id, [r])
        print(f"Inserted {len(rows)} opportunity outcome rows into {table_id}")
    else:
        print('No opportunity outcome rows computed')


if __name__ == '__main__':
    main()

