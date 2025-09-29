#!/usr/bin/env python3
"""
Build high-intent breakdown by channel (source/medium) from GA4 export.

Output: <project>.<dataset>.ga4_high_intent_by_channel_daily
  - date, source, medium, enrollment_loaded, begin_checkout, form_submit_enroll, high_intent_no_purchase_7

Notes:
  - Uses daily user-level intent flags and picks the latest non-null source/medium cookie seen that day per user.
  - Excludes users who purchase within post_trial_window_days (from mapping) after the intent day.
"""
import os
from datetime import date, timedelta
from google.cloud import bigquery
import yaml


def load_mapping(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    mapping_path = os.environ.get('GA4_MAPPING_PATH', 'AELP2/config/ga4_mapping.yaml')
    cfg = load_mapping(mapping_path)
    trial_regex = cfg['trial_page_regex']
    win = int(cfg.get('post_trial_window_days', 7))
    days = int(os.getenv('GA4_AUDIENCE_DAYS', '14'))
    start = (date.today() - timedelta(days=days)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')

    c = bigquery.Client(project=project)
    sql = f"""
    WITH events AS (
      SELECT
        PARSE_DATE('%Y%m%d', e.event_date) AS date,
        e.user_pseudo_id,
        e.event_timestamp,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='source_cookie') AS source,
        (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='medium_cookie') AS medium,
        EXISTS(SELECT 1 FROM UNNEST(e.event_params) ep WHERE ep.key='enrollment_loaded') AS enroll_loaded,
        (e.event_name IN ('begin_checkout','Checkout')) AS began_checkout,
        (e.event_name='form_submit' AND EXISTS(
           SELECT 1 FROM UNNEST(e.event_params) ep WHERE ep.key='page_location' AND REGEXP_CONTAINS(ep.value.string_value, @trial_regex)
        )) AS form_submit_enroll
      FROM `{export_ds}.events_*` e
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
    ), ranked AS (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY date, user_pseudo_id ORDER BY event_timestamp DESC) AS rn
      FROM events
      WHERE source IS NOT NULL OR medium IS NOT NULL
    ), pick AS (
      SELECT date, user_pseudo_id, source, medium
      FROM ranked
      WHERE rn=1
    ), daily AS (
      SELECT e.date, e.user_pseudo_id,
             MAX(IF(e.enroll_loaded,1,0)) AS enroll_loaded,
             MAX(IF(e.began_checkout,1,0)) AS begin_checkout,
             MAX(IF(e.form_submit_enroll,1,0)) AS form_submit_enroll,
             ANY_VALUE(p.source) AS source,
             ANY_VALUE(p.medium) AS medium
      FROM events e
      LEFT JOIN pick p USING(date, user_pseudo_id)
      GROUP BY e.date, e.user_pseudo_id
    ), purchases AS (
      SELECT PARSE_DATE('%Y%m%d', event_date) AS pdate, user_pseudo_id
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' AND event_name='purchase'
    ), no_purchase AS (
      SELECT d.date, d.user_pseudo_id,
             (d.enroll_loaded=1 OR d.begin_checkout=1 OR d.form_submit_enroll=1) AS is_hi,
             IFNULL(d.source,'(none)') AS source,
             IFNULL(d.medium,'(none)') AS medium,
             NOT EXISTS (
               SELECT 1 FROM purchases p
               WHERE p.user_pseudo_id = d.user_pseudo_id
                 AND p.pdate BETWEEN d.date AND DATE_ADD(d.date, INTERVAL @win DAY)
             ) AS no_purchase_7
      FROM daily d
    )
    SELECT date, source, medium,
           SUM(enroll_loaded) AS enrollment_loaded,
           SUM(begin_checkout) AS begin_checkout,
           SUM(form_submit_enroll) AS form_submit_enroll,
           COUNTIF(is_hi AND no_purchase_7) AS high_intent_no_purchase_7
    FROM (
      SELECT d.date, IFNULL(d.source,'(none)') AS source, IFNULL(d.medium,'(none)') AS medium,
             d.enroll_loaded, d.begin_checkout, d.form_submit_enroll
      FROM daily d
    ) d
    JOIN no_purchase n USING(date, source, medium)
    GROUP BY date, source, medium
    ORDER BY date, high_intent_no_purchase_7 DESC
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('trial_regex','STRING', trial_regex),
            bigquery.ScalarQueryParameter('win','INT64', win),
        ]
    )
    rows = [dict(r) for r in c.query(sql, job_config=job_config, location='US').result()]
    for r in rows:
        r['date'] = r['date'].strftime('%Y-%m-%d')
        for k in ('enrollment_loaded','begin_checkout','form_submit_enroll','high_intent_no_purchase_7'):
            r[k] = int(r.get(k) or 0)

    table_id = f"{project}.{dataset}.ga4_high_intent_by_channel_daily"
    schema = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('source','STRING'),
        bigquery.SchemaField('medium','STRING'),
        bigquery.SchemaField('enrollment_loaded','INT64'),
        bigquery.SchemaField('begin_checkout','INT64'),
        bigquery.SchemaField('form_submit_enroll','INT64'),
        bigquery.SchemaField('high_intent_no_purchase_7','INT64'),
    ]
    c.delete_table(table_id, not_found_ok=True)
    t = bigquery.Table(table_id, schema=schema)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t)
    job = c.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()
    print('Built ga4_high_intent_by_channel_daily')


if __name__ == '__main__':
    main()
