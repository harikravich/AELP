#!/usr/bin/env python3
"""
Build touch-aligned (delayed-reward) daily series from GA4 export.

Outputs in <project>.<dataset>:
  - ga_triggered_channel_daily(date, channel, triggered_conversions)
  - ga_triggered_daily(date, triggered_conversions)

Method
  - For each purchase, collect prior touches in a 14-day window.
  - Apply exponential time-decay weights (half-life = 7 days) by touch age.
  - Normalize weights to sum to 1.0 per purchase (fractional conversion credit).
  - Attribute fractional credit to each touch's channel and touch date.

Env:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  - GA4_EXPORT_DATASET (default: ga360-bigquery-datashare.analytics_308028264)
  - ATTRIB_LOOKBACK_DAYS (default 14), ATTRIB_HALFLIFE_DAYS (default 7)
"""
import os
from datetime import date, timedelta
from typing import Dict, Any, List

from google.cloud import bigquery


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    lookback = int(os.getenv('ATTRIB_LOOKBACK_DAYS', '14'))
    halflife = int(os.getenv('ATTRIB_HALFLIFE_DAYS', '7'))

    c = bigquery.Client(project=project)

    # SQL: touches in window with their dates; purchases with conversion date; join and compute age.
    # Improvements:
    #  - Fallback to traffic_source.source/medium when cookie params are missing
    #  - Include common touch events (session_start/page_view/etc.)
    #  - Add a "direct" fallback for purchases with no prior touch in window
    sql = f"""
    WITH touches AS (
      SELECT
        user_pseudo_id,
        TIMESTAMP_MICROS(event_timestamp) AS ts,
        DATE(TIMESTAMP_MICROS(event_timestamp)) AS touch_date,
        COALESCE((SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='source_cookie'), traffic_source.source, '(none)') AS src,
        COALESCE((SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='medium_cookie'), traffic_source.medium, '(none)') AS med
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY))
                              AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
        AND event_name IN ('session_start','page_view','view_item','begin_checkout','add_to_cart','purchase')
    ), purchases AS (
      SELECT user_pseudo_id,
             TIMESTAMP_MICROS(event_timestamp) AS p_ts,
             DATE(TIMESTAMP_MICROS(event_timestamp)) AS conv_date
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL 120 DAY))
                              AND FORMAT_DATE('%Y%m%d', CURRENT_DATE())
        AND event_name='purchase'
    ), joined AS (
      SELECT p.user_pseudo_id,
             p.p_ts,
             p.conv_date,
             t.touch_date,
             CONCAT(t.src,'/',t.med) AS channel,
             DATE_DIFF(p.conv_date, t.touch_date, DAY) AS age
      FROM purchases p
      JOIN touches t
        ON t.user_pseudo_id = p.user_pseudo_id
       AND t.ts BETWEEN TIMESTAMP_SUB(p.p_ts, INTERVAL {lookback} DAY) AND p.p_ts
    ), weighted AS (
      SELECT user_pseudo_id, p_ts, conv_date, touch_date, channel, age,
             POW(0.5, age/{halflife}) AS w
      FROM joined
      WHERE age BETWEEN 0 AND {lookback}
    ), norm AS (
      SELECT user_pseudo_id, p_ts, conv_date, touch_date, channel,
             SAFE_DIVIDE(w, SUM(w) OVER (PARTITION BY user_pseudo_id, p_ts)) AS frac
      FROM weighted
    ), orphans AS (
      -- Purchases with no prior touch in window → credit to (direct)/(none) on conv_date
      SELECT p.user_pseudo_id, p.p_ts, p.conv_date AS touch_date, '(direct)/(none)' AS channel, 1.0 AS frac
      FROM purchases p
      WHERE NOT EXISTS (
        SELECT 1 FROM touches t
        WHERE t.user_pseudo_id = p.user_pseudo_id
          AND t.ts BETWEEN TIMESTAMP_SUB(p.p_ts, INTERVAL {lookback} DAY) AND p.p_ts
      )
    )
    SELECT date, channel, SUM(frac) AS triggered_conversions
    FROM (
      SELECT touch_date AS date, channel, frac FROM norm
      UNION ALL
      SELECT touch_date AS date, channel, frac FROM orphans
    )
    GROUP BY date, channel
    """

    rows = [dict(r) for r in c.query(sql, location='US').result()]
    # Write channel-level table
    schema_chan = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('channel','STRING','REQUIRED'),
        bigquery.SchemaField('triggered_conversions','FLOAT','REQUIRED'),
    ]
    chan_id = f"{project}.{dataset}.ga_triggered_channel_daily"
    try:
        c.delete_table(chan_id, not_found_ok=True)
    except Exception:
        pass
    t = bigquery.Table(chan_id, schema=schema_chan)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t)
    # Normalize date to ISO string for JSON load
    rows_json = []
    for r in rows:
        rows_json.append({'date': r['date'].isoformat() if hasattr(r['date'], 'isoformat') else r['date'],
                          'channel': r['channel'],
                          'triggered_conversions': float(r['triggered_conversions'] or 0.0)})
    c.load_table_from_json(rows_json, chan_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()

    # Aggregate to total
    agg = {}
    for r in rows:
        d = r['date']
        agg[d] = agg.get(d, 0.0) + float(r['triggered_conversions'] or 0.0)
    rows2 = [{'date': d.isoformat() if hasattr(d, 'isoformat') else d, 'triggered_conversions': float(v)} for d, v in agg.items()]
    schema_tot = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('triggered_conversions','FLOAT','REQUIRED'),
    ]
    tot_id = f"{project}.{dataset}.ga_triggered_daily"
    try:
        c.delete_table(tot_id, not_found_ok=True)
    except Exception:
        pass
    t2 = bigquery.Table(tot_id, schema=schema_tot)
    t2.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t2)
    c.load_table_from_json(rows2, tot_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f"Wrote {len(rows)} channel rows and {len(rows2)} total rows")


if __name__ == '__main__':
    main()
