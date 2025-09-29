#!/usr/bin/env python3
"""
Uplift v1: Bootstrap segment uplift evaluation from journey tables (if present).

Reads `gaelp_users.persistent_touchpoints` joined to simple conversion flags and
computes per-segment exposure rates and conversion rates for exposed vs. unexposed cohorts.
Writes results to `<project>.<dataset>.uplift_segment_daily`.

If journey tables are missing, creates the output table and exits gracefully.
"""
import os
from datetime import date, timedelta, datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_table(bq: bigquery.Client, project: str, dataset: str):
    table_id = f"{project}.{dataset}.uplift_segment_daily"
    try:
        bq.get_table(table_id)
        return
    except NotFound:
        pass
    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("segment", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("exposed", "BOOL", mode="REQUIRED"),
        bigquery.SchemaField("users", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("conversions", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("conversion_rate", "FLOAT", mode="NULLABLE"),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="date")
    bq.create_table(table)


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    bq = bigquery.Client(project=project)
    ensure_table(bq, project, dataset)

    # Check journey tables
    users_ds = f"{project}.gaelp_users"
    try:
        bq.get_table(f"{users_ds}.persistent_touchpoints")
        bq.get_table(f"{users_ds}.journey_sessions")
    except NotFound:
        print('Journey tables not found; uplift table ensured, nothing to compute')
        return

    # Simple uplift proxy: compare conversion rates for users with any ad exposure vs none, by segment
    # Assumptions: journey_sessions has user_id, segment; persistent_touchpoints has user_id and channel
    sql = f"""
      WITH exposed AS (
        SELECT js.user_id, js.segment
        FROM `{users_ds}.journey_sessions` js
        WHERE EXISTS (
          SELECT 1 FROM `{users_ds}.persistent_touchpoints` tp
          WHERE tp.user_id = js.user_id
        )
      ),
      unexposed AS (
        SELECT js.user_id, js.segment
        FROM `{users_ds}.journey_sessions` js
        WHERE NOT EXISTS (
          SELECT 1 FROM `{users_ds}.persistent_touchpoints` tp
          WHERE tp.user_id = js.user_id
        )
      ),
      conversions AS (
        SELECT js.user_id, 1 AS converted
        FROM `{users_ds}.journey_sessions` js
        WHERE js.converted = TRUE
      )
      SELECT CURRENT_DATE() AS date, seg.segment AS segment, seg.exposed AS exposed,
             COUNT(*) AS users,
             SUM(IF(c.converted IS NULL, 0, 1)) AS conversions,
             SAFE_DIVIDE(SUM(IF(c.converted IS NULL, 0, 1)), COUNT(*)) AS conversion_rate
      FROM (
        SELECT user_id, segment, TRUE AS exposed FROM exposed
        UNION ALL
        SELECT user_id, segment, FALSE AS exposed FROM unexposed
      ) seg
      LEFT JOIN conversions c ON c.user_id = seg.user_id
      GROUP BY date, segment, exposed
    """

    rows = list(bq.query(sql).result())
    if not rows:
        print('No uplift rows produced (empty journeys)')
        return
    table_id = f"{project}.{dataset}.uplift_segment_daily"
    out = []
    for r in rows:
        out.append({
            'date': r['date'].isoformat(),
            'segment': r['segment'],
            'exposed': bool(r['exposed']),
            'users': int(r['users'] or 0),
            'conversions': int(r['conversions'] or 0),
            'conversion_rate': float(r['conversion_rate'] or 0.0),
        })
    bq.insert_rows_json(table_id, out)
    print(f"Wrote {len(out)} uplift rows to {table_id}")


if __name__ == '__main__':
    main()
