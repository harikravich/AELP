#!/usr/bin/env python3
"""
Journeys Populate (bootstrap)

Ensures gaelp_users.journey_sessions and persistent_touchpoints exist.
If GA4 export dataset is available (env GA4_EXPORT_DATASET), extracts a minimal
set of sessions and touchpoints for the last N days and writes sample rows.

Safe to run repeatedly; no-ops if GA4 export isn’t configured.
"""
import os
from datetime import date, timedelta
from google.cloud import bigquery
from google.cloud.exceptions import NotFound


def ensure_tables(bq: bigquery.Client, project: str):
    ds = f"{project}.gaelp_users"
    for name, schema in [
        ('journey_sessions', [
            bigquery.SchemaField('user_id', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('session_start', 'TIMESTAMP'),
            bigquery.SchemaField('segment', 'STRING'),
            bigquery.SchemaField('converted', 'BOOL'),
        ]),
        ('persistent_touchpoints', [
            bigquery.SchemaField('user_id', 'STRING', mode='REQUIRED'),
            bigquery.SchemaField('timestamp', 'TIMESTAMP'),
            bigquery.SchemaField('channel', 'STRING'),
            bigquery.SchemaField('campaign_id', 'STRING'),
            bigquery.SchemaField('metadata', 'JSON'),
        ]),
    ]:
        table_id = f"{ds}.{name}"
        try:
            bq.get_table(table_id)
        except NotFound:
            t = bigquery.Table(table_id, schema=schema)
            t.time_partitioning = bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field=schema[1].name)
            bq.create_table(t)


def populate_from_ga4(bq: bigquery.Client, project: str, export_dataset: str, days: int = 7):
    ds = f"{project}.gaelp_users"
    # Minimal session list (if GA4 export exists). This is illustrative; production joins would be richer.
    sql_sessions = f"""
      SELECT CONCAT('u_', CAST(FARM_FINGERPRINT(user_pseudo_id) AS STRING)) AS user_id,
             TIMESTAMP_MICROS(MIN(event_timestamp)) AS session_start,
             'unknown' AS segment,
             FALSE AS converted
      FROM `{project}.{export_dataset}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)) AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
        AND event_name = 'session_start'
      GROUP BY user_pseudo_id
      LIMIT 1000
    """
    rows = [dict(r) for r in bq.query(sql_sessions).result()]
    if rows:
        bq.insert_rows_json(f"{ds}.journey_sessions", rows)
    # Minimal touchpoints stub (empty for now)


def main():
    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project:
        raise RuntimeError('Set GOOGLE_CLOUD_PROJECT')
    bq = bigquery.Client(project=project)
    ensure_tables(bq, project)
    export_ds = os.getenv('GA4_EXPORT_DATASET')
    if export_ds:
        try:
            populate_from_ga4(bq, project, export_ds)
            print('Journeys populated from GA4 export (sample).')
        except Exception as e:
            print(f'Journeys GA4 populate skipped: {e}')
    else:
        print('GA4_EXPORT_DATASET not set; ensured journey tables only.')


if __name__ == '__main__':
    main()

