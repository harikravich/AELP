#!/usr/bin/env python3
import os
from datetime import date, timedelta
from google.cloud import bigquery
import yaml


def load_mapping(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_table(c: bigquery.Client, table_id: str):
    from google.cloud import bigquery as bq
    try:
        c.get_table(table_id)
        return
    except Exception:
        schema = [
            bq.SchemaField('date','DATE','REQUIRED'),
            bq.SchemaField('source','STRING'),
            bq.SchemaField('medium','STRING'),
            bq.SchemaField('enrollment_loaded','INT64'),
            bq.SchemaField('begin_checkout','INT64'),
            bq.SchemaField('form_submit_enroll','INT64'),
            bq.SchemaField('high_intent_no_purchase_7','INT64'),
        ]
        t = bq.Table(table_id, schema=schema)
        t.time_partitioning = bq.TimePartitioning(field='date')
        c.create_table(t)


def build_one_day(c: bigquery.Client, export_ds: str, table_id: str, day: str, trial_regex: str, win: int):
    from google.cloud import bigquery as bq
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
      WHERE _TABLE_SUFFIX=@day
    ), ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY date, user_pseudo_id ORDER BY event_timestamp DESC) AS rn
      FROM events
      WHERE source IS NOT NULL OR medium IS NOT NULL
    ), pick AS (
      SELECT date, user_pseudo_id, source, medium FROM ranked WHERE rn=1
    ), daily AS (
      SELECT e.date, e.user_pseudo_id,
             MAX(IF(e.enroll_loaded,1,0)) AS enrollment_loaded,
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
      WHERE _TABLE_SUFFIX BETWEEN @day AND FORMAT_DATE('%Y%m%d', DATE_ADD(PARSE_DATE('%Y%m%d',@day), INTERVAL @win DAY))
        AND event_name='purchase'
    ), intent AS (
      SELECT d.date, d.user_pseudo_id,
             (d.enrollment_loaded=1 OR d.begin_checkout=1 OR d.form_submit_enroll=1) AS is_hi,
             IFNULL(d.source,'(none)') AS source,
             IFNULL(d.medium,'(none)') AS medium
      FROM daily d
    ), no_purchase AS (
      SELECT i.date, i.user_pseudo_id, i.source, i.medium,
             NOT EXISTS (
               SELECT 1 FROM purchases p
               WHERE p.user_pseudo_id = i.user_pseudo_id
                 AND p.pdate BETWEEN i.date AND DATE_ADD(i.date, INTERVAL @win DAY)
             ) AS no_purchase_7
      FROM intent i
    )
    SELECT d.date,
           IFNULL(d.source,'(none)') AS source,
           IFNULL(d.medium,'(none)') AS medium,
           SUM(d.enrollment_loaded) AS enrollment_loaded,
           SUM(d.begin_checkout) AS begin_checkout,
           SUM(d.form_submit_enroll) AS form_submit_enroll,
           COUNTIF((d.enrollment_loaded=1 OR d.begin_checkout=1 OR d.form_submit_enroll=1) AND n.no_purchase_7) AS high_intent_no_purchase_7
    FROM daily d
    JOIN no_purchase n
      ON n.date=d.date
     AND n.source=IFNULL(d.source,'(none)')
     AND n.medium=IFNULL(d.medium,'(none)')
    GROUP BY d.date, source, medium
    """
    job_config = bq.QueryJobConfig(
        query_parameters=[
            bq.ScalarQueryParameter('trial_regex','STRING', trial_regex),
            bq.ScalarQueryParameter('win','INT64', win),
            bq.ScalarQueryParameter('day','STRING', day),
        ]
    )
    rows = [dict(r) for r in c.query(sql, job_config=job_config, location='US').result()]
    # normalize
    for r in rows:
        r['date'] = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"
        for k in ('enrollment_loaded','begin_checkout','form_submit_enroll','high_intent_no_purchase_7'):
            r[k] = int(r.get(k) or 0)
    if not rows:
        return 0
    job = c.load_table_from_json(rows, table_id,
                                 job_config=bq.LoadJobConfig(write_disposition='WRITE_APPEND'))
    job.result()
    return len(rows)


def run_chunked():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    mapping_path = os.environ.get('GA4_MAPPING_PATH', 'AELP2/config/ga4_mapping.yaml')
    cfg = load_mapping(mapping_path)
    trial_regex = cfg['trial_page_regex']
    win = int(cfg.get('post_trial_window_days', 7))
    days = int(os.getenv('GA4_AUDIENCE_DAYS', '14'))
    c = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.ga4_high_intent_by_channel_daily"
    ensure_table(c, table_id)

    total = 0
    for i in range(days, 0, -1):
        day = (date.today() - timedelta(days=i)).strftime('%Y%m%d')
        cnt = build_one_day(c, export_ds, table_id, day, trial_regex, win)
        total += cnt
        print(f"[chunk] {day}: {cnt} rows appended")
    print(f"[done] appended rows: {total}")


if __name__ == '__main__':
    run_chunked()
