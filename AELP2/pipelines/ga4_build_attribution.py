#!/usr/bin/env python3
"""
Build GA attribution daily tables from export using purchase-level cookies (first pass LND proxy):
  - ga4_attribution_daily(date, source, medium, purchases, initial_payment_revenue)

Notes:
  - Uses purchase event's source_cookie/medium_cookie as last-non-direct proxy.
  - Adds host (landing host) for debugging.
"""
import os
from datetime import date, timedelta
from google.cloud import bigquery


def main():
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    days = int(os.getenv('GA4_ATTRIB_DAYS', '120'))
    start = (date.today() - timedelta(days=days)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')
    intraday_end = f"intraday_{end}"
    prev = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
    intraday_prev = f"intraday_{prev}"
    suffix_pred = f"(_TABLE_SUFFIX BETWEEN '{start}' AND '{end}' OR _TABLE_SUFFIX IN ('{intraday_prev}','{intraday_end}'))"
    c = bigquery.Client(project=project)
    sql = f"""
    WITH p AS (
      SELECT PARSE_DATE('%Y%m%d', e.event_date) AS date,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='source_cookie') AS source,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='medium_cookie') AS medium,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='landing_page_cookie') AS landing,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='initial_payment_revenue') AS init_rev
      FROM `{export_ds}.events_*` e
      WHERE {suffix_pred} AND e.event_name='purchase'
    )
    SELECT date,
           IFNULL(source,'(none)') AS source,
           IFNULL(medium,'(none)') AS medium,
           SAFE_CAST(REGEXP_EXTRACT(landing, r"https?://([^/]+)") AS STRING) AS host,
           COUNT(1) AS purchases,
           SUM(SAFE_CAST(init_rev AS FLOAT64)) AS initial_payment_revenue
    FROM p
    GROUP BY date, source, medium, host
    ORDER BY date, purchases DESC
    """
    rows = [dict(r) for r in c.query(sql, location='US').result()]
    for r in rows:
        r['date'] = r['date'].strftime('%Y-%m-%d')
        r['purchases'] = int(r['purchases'] or 0)
        r['initial_payment_revenue'] = float(r['initial_payment_revenue'] or 0.0)

    table_id = f"{project}.{dataset}.ga4_attribution_daily"
    schema = [
        bigquery.SchemaField('date','DATE','REQUIRED'),
        bigquery.SchemaField('source','STRING'),
        bigquery.SchemaField('medium','STRING'),
        bigquery.SchemaField('host','STRING'),
        bigquery.SchemaField('purchases','INT64'),
        bigquery.SchemaField('initial_payment_revenue','FLOAT64'),
    ]
    c.delete_table(table_id, not_found_ok=True)
    t = bigquery.Table(table_id, schema=schema)
    t.time_partitioning = bigquery.TimePartitioning(field='date')
    c.create_table(t)
    job = c.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()
    print('Built ga4_attribution_daily')


if __name__ == '__main__':
    main()
