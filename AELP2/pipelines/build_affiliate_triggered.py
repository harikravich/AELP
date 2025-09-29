#!/usr/bin/env python3
"""
Affiliate-triggered (delayed-reward) series from GA4 export.

Outputs in <project>.<dataset>:
  - ga_affiliate_triggered_channel_daily(date, channel, triggered_conversions)
  - ga_affiliate_triggered_daily(date, triggered_conversions)

Affiliate touch heuristic:
  - medium contains 'affiliate' OR
  - source matches regex AFFIL_REGEX (default covers common networks & coupon/loyalty keywords)

Env:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET
  - GA4_EXPORT_DATASET (default: ga360-bigquery-datashare.analytics_308028264)
  - ATTRIB_LOOKBACK_DAYS (default 14), ATTRIB_HALFLIFE_DAYS (default 7)
  - AFFIL_REGEX (default: '(impact|rakuten|cj|awin|partnerize|shareasale|refersion|linkconnector|flexoffers|pepperjam|honey|retailmenot|slickdeals|coupon|deal|cashback|rebate|loyalty|ebates|prodege)')
"""
from __future__ import annotations
import os
from google.cloud import bigquery


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    lookback = int(os.getenv('ATTRIB_LOOKBACK_DAYS', '14'))
    halflife = int(os.getenv('ATTRIB_HALFLIFE_DAYS', '7'))
    affil_regex = os.getenv('AFFIL_REGEX', '(impact|rakuten|cj|awin|partnerize|shareasale|refersion|linkconnector|flexoffers|pepperjam|honey|retailmenot|slickdeals|coupon|deal|cashback|rebate|loyalty|ebates|prodege)')

    c = bigquery.Client(project=project)
    sql = f"""
    DECLARE affil_regex STRING DEFAULT @affil_regex;
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
        AND (
          LOWER(COALESCE((SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='medium_cookie'), traffic_source.medium, '(none)')) LIKE '%affiliate%'
          OR REGEXP_CONTAINS(LOWER(traffic_source.source), affil_regex)
          OR REGEXP_CONTAINS(LOWER(COALESCE((SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(event_params) ep WHERE ep.key='source_cookie'), traffic_source.source, '(none)')), affil_regex)
        )
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
    )
    SELECT touch_date AS date, channel, SUM(frac) AS triggered_conversions
    FROM norm
    GROUP BY date, channel
    """

    rows = [dict(r) for r in c.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[bigquery.ScalarQueryParameter('affil_regex','STRING', affil_regex)]), location='US').result()]

    # Write channel table
    chan_id = f"{project}.{dataset}.ga_affiliate_triggered_channel_daily"
    schema_c = [bigquery.SchemaField('date','DATE','REQUIRED'),
                bigquery.SchemaField('channel','STRING','REQUIRED'),
                bigquery.SchemaField('triggered_conversions','FLOAT','REQUIRED')]
    try:
        c.delete_table(chan_id, not_found_ok=True)
    except Exception:
        pass
    t = bigquery.Table(chan_id, schema=schema_c)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t)
    rows_json = [{'date': (r['date'].isoformat() if hasattr(r['date'],'isoformat') else r['date']), 'channel': r['channel'], 'triggered_conversions': float(r['triggered_conversions'] or 0.0)} for r in rows]
    c.load_table_from_json(rows_json, chan_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()

    # Total
    agg = {}
    for r in rows:
        d = r['date']
        agg[d] = agg.get(d, 0.0) + float(r['triggered_conversions'] or 0.0)
    tot_id = f"{project}.{dataset}.ga_affiliate_triggered_daily"
    schema_t = [bigquery.SchemaField('date','DATE','REQUIRED'), bigquery.SchemaField('triggered_conversions','FLOAT','REQUIRED')]
    try:
        c.delete_table(tot_id, not_found_ok=True)
    except Exception:
        pass
    t2 = bigquery.Table(tot_id, schema=schema_t)
    t2.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t2)
    rows2 = [{'date': (d.isoformat() if hasattr(d,'isoformat') else d), 'triggered_conversions': float(v)} for d,v in agg.items()]
    c.load_table_from_json(rows2, tot_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')).result()
    print(f"Wrote {len(rows)} affiliate channel rows and {len(rows2)} affiliate total rows")


if __name__ == '__main__':
    main()
