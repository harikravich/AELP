#!/usr/bin/env python3
"""
Build high-intent audiences from GA4 export:
  - ga4_high_intent_daily: date, enrollment_loaded, begin_checkout, form_submit_enroll, high_intent_no_purchase_7

Uses:
  - enrollment_loaded param presence
  - begin_checkout / Checkout events
  - form_submit on enrollment/onboarding pages
  - Excludes users who purchase within N days (default 7)
"""
import os
from datetime import date, timedelta
from typing import Dict, Any
from google.cloud import bigquery
import yaml


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    mapping_path = os.environ.get('GA4_MAPPING_PATH', 'AELP2/config/ga4_mapping.yaml')
    cfg = load_mapping(mapping_path)
    win = int(cfg.get('post_trial_window_days', 7))
    trial_regex = cfg['trial_page_regex']
    days_back = int(os.getenv('GA4_AUDIENCE_DAYS', 120))
    start = (date.today() - timedelta(days=days_back)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')

    client = bigquery.Client(project=project)

    # Optimized single-pass daily aggregation (limits event_param unnests to subqueries)
    sql = f"""
    WITH daily AS (
      SELECT
        PARSE_DATE('%Y%m%d', e.event_date) AS date,
        e.user_pseudo_id,
        MAX(IF(e.event_name IN ('begin_checkout','Checkout'), 1, 0)) AS began_checkout,
        MAX(IF((SELECT COUNT(1) FROM UNNEST(e.event_params) ep WHERE ep.key='enrollment_loaded') > 0, 1, 0)) AS enroll_loaded,
        MAX(IF(
          e.event_name='form_submit' AND EXISTS (
            SELECT 1 FROM UNNEST(e.event_params) ep
            WHERE ep.key='page_location' AND REGEXP_CONTAINS(ep.value.string_value, @trial_regex)
          ), 1, 0)) AS form_submit_enroll
      FROM `{export_ds}.events_*` e
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
      GROUP BY date, e.user_pseudo_id
    ), purchases AS (
      SELECT PARSE_DATE('%Y%m%d', event_date) AS pdate, user_pseudo_id
      FROM `{export_ds}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' AND event_name='purchase'
    ), intent AS (
      SELECT d.date, d.user_pseudo_id,
             (d.enroll_loaded=1 OR d.began_checkout=1 OR d.form_submit_enroll=1) AS is_hi
      FROM daily d
    ), intent_agg AS (
      SELECT date,
             SUM(IF(enroll_loaded=1,1,0)) AS enrollment_loaded,
             SUM(IF(began_checkout=1,1,0)) AS begin_checkout,
             SUM(IF(form_submit_enroll=1,1,0)) AS form_submit_enroll
      FROM daily
      GROUP BY date
    ), hi_no_purchase AS (
      SELECT i.date,
             COUNTIF(i.is_hi AND NOT EXISTS (
               SELECT 1 FROM purchases p
               WHERE p.user_pseudo_id = i.user_pseudo_id
                 AND p.pdate BETWEEN i.date AND DATE_ADD(i.date, INTERVAL @win DAY)
             )) AS high_intent_no_purchase_7
      FROM intent i
      GROUP BY i.date
    )
    SELECT a.date, a.enrollment_loaded, a.begin_checkout, a.form_submit_enroll, n.high_intent_no_purchase_7
    FROM intent_agg a
    JOIN hi_no_purchase n USING(date)
    ORDER BY a.date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter('trial_regex', 'STRING', trial_regex),
            bigquery.ScalarQueryParameter('win', 'INT64', win),
        ]
    )
    rows = [dict(r) for r in client.query(sql, job_config=job_config, location='US').result()]
    for r in rows:
        r['date'] = r['date'].strftime('%Y-%m-%d')
        for k in ('enrollment_loaded','begin_checkout','form_submit_enroll','high_intent_no_purchase_7'):
            r[k] = int(r.get(k) or 0)

    table_id = f"{project}.{dataset}.ga4_high_intent_daily"
    schema = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('enrollment_loaded','INT64'),
        bigquery.SchemaField('begin_checkout','INT64'),
        bigquery.SchemaField('form_submit_enroll','INT64'),
        bigquery.SchemaField('high_intent_no_purchase_7','INT64'),
    ]
    client.delete_table(table_id, not_found_ok=True)
    t = bigquery.Table(table_id, schema=schema)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    client.create_table(t)
    job = client.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()
    print('Built ga4_high_intent_daily')


if __name__ == '__main__':
    main()
