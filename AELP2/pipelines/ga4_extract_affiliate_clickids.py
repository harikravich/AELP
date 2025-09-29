#!/usr/bin/env python3
"""
Extract affiliate click IDs from GA4 export into a joinable table.

Output: <project>.<dataset>.ga_affiliate_clickids
Fields: date DATE, event_ts TIMESTAMP, user_pseudo_id STRING, click_id STRING, source STRING, medium STRING

We parse clickid/irclickid/ir_clickid from page_location and from event_params.
"""
import os
from google.cloud import bigquery

def main():
    project=os.getenv('GOOGLE_CLOUD_PROJECT'); dataset=os.getenv('BIGQUERY_TRAINING_DATASET')
    if not project or not dataset:
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT and BIGQUERY_TRAINING_DATASET')
    client=bigquery.Client(project=project)
    ds=os.getenv('GA4_EXPORT_DATASET','ga360-bigquery-datashare.analytics_308028264')
    table=f"{project}.{dataset}.ga_affiliate_clickids"
    client.delete_table(table, not_found_ok=True)
    client.create_table(bigquery.Table(table, schema=[
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('event_ts','TIMESTAMP','REQUIRED'),
        bigquery.SchemaField('user_pseudo_id','STRING','REQUIRED'),
        bigquery.SchemaField('click_id','STRING','REQUIRED'),
        bigquery.SchemaField('source','STRING'),
        bigquery.SchemaField('medium','STRING'),
    ]))
    sql=f"""
    WITH base AS (
      SELECT
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS date,
        TIMESTAMP_MICROS(event_timestamp) AS event_ts,
        user_pseudo_id,
        traffic_source.source AS source,
        traffic_source.medium AS medium,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='page_location') AS page_location,
        ARRAY(SELECT AS STRUCT ep.key, ep.value.string_value AS sv FROM UNNEST(event_params) ep) AS ev
      FROM `{ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY))
                              AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
        AND event_name IN ('page_view','session_start','purchase')
    ), parsed AS (
      SELECT date, event_ts, user_pseudo_id, source, medium,
        COALESCE(
          REGEXP_EXTRACT(page_location, r'[?&](?:irclickid|ir_clickid|clickid)=([^&#]+)'),
          (SELECT ANY_VALUE(sv) FROM UNNEST(ev) WHERE key IN ('irclickid','ir_clickid','clickid'))
        ) AS click_id
      FROM base
    )
    SELECT date, event_ts, user_pseudo_id, click_id, source, medium
    FROM parsed
    WHERE click_id IS NOT NULL
    """
    rows=[dict(r) for r in client.query(sql, location='US').result()]
    # Load in chunks
    if rows:
        for i in range(0, len(rows), 5000):
            client.insert_rows_json(table, rows[i:i+5000])
    print(f"ga_affiliate_clickids: {len(rows)} rows")

if __name__=='__main__':
    main()

