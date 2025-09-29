#!/usr/bin/env python3
"""
Build GA-derived daily/monthly KPI tables and cohort summaries from GA4 export (events_*).

Outputs (in <project>.<dataset>):
  - ga4_derived_daily(date, enrollments, d2p_starts, post_trial_subs, mobile_subs)
  - ga4_derived_monthly(month, enrollments, d2p_starts, post_trial_subs, mobile_subs)
  - ga4_offer_code_daily(date, plan_code, cc, source, medium, host, purchases, revenue)

Config: AELP2/config/ga4_mapping.yaml

Notes:
  - Reads GA4 export in US location; writes to training dataset (likely us-central1).
  - Uses server-side aggregation queries; fetches aggregated rows and streams to BQ.
"""

import os
import sys
import yaml
from datetime import date, timedelta
from typing import Dict, Any, List, Tuple
from google.cloud import bigquery


def load_mapping(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_clients() -> Tuple[bigquery.Client, str, str, str]:
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    dataset = os.environ['BIGQUERY_TRAINING_DATASET']
    export_ds = os.environ.get('GA4_EXPORT_DATASET', 'ga360-bigquery-datashare.analytics_308028264')
    client = bigquery.Client(project=project)
    return client, project, dataset, export_ds


def ensure_table(client: bigquery.Client, table_id: str, schema: List[bigquery.SchemaField]) -> None:
    try:
        client.get_table(table_id)
    except Exception:
        t = bigquery.Table(table_id, schema=schema)
        if any(f.name == 'date' and f.field_type == 'DATE' for f in schema):
            t.time_partitioning = bigquery.TimePartitioning(field='date')
        client.create_table(t)


def query(client: bigquery.Client, sql: str, location: str = 'US'):
    return client.query(sql, location=location).result()

def query_params(client: bigquery.Client, sql: str, params: Dict[str, Any], location: str = 'US'):
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter(k, 'STRING', v) for k, v in params.items()]
    )
    return client.query(sql, job_config=job_config, location=location).result()


def build_daily(client: bigquery.Client, project: str, dataset: str, export_ds: str, cfg: Dict[str, Any]) -> None:
    days_back = int(os.getenv('GA4_DERIVED_DAYS', cfg.get('days_back', 450)))
    start = (date.today() - timedelta(days=days_back)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')
    # Suffix filter that also includes intraday table for the last day to avoid zeros on current day
    suffix_pred = f"(_TABLE_SUFFIX BETWEEN '{start}' AND '{end}' OR _TABLE_SUFFIX = '20250916')"  # placeholder replaced below
    # Build suffix predicate generically: BETWEEN for historical + intraday_<end> if present
    intraday_end = f"intraday_{end}"
    prev = (date.today() - timedelta(days=1)).strftime('%Y%m%d')
    intraday_prev = f"intraday_{prev}"
    suffix_pred = (
        f"(_TABLE_SUFFIX BETWEEN '{start}' AND '{end}' OR _TABLE_SUFFIX IN ('{intraday_prev}','{intraday_end}'))"
    )
    trial_regex = cfg['trial_page_regex']
    post_win = int(cfg.get('post_trial_window_days', 7))
    d2p_look = int(cfg.get('d2p_lookback_days', 14))
    mobile = cfg.get('mobile_rules', {})
    cc_re = mobile.get('cc_regex', '')
    plan_re = mobile.get('plan_code_regex', '')

    # 1) Enrollments (purchase) per day
    q_enr = f"""
    SELECT PARSE_DATE('%Y%m%d', event_date) AS date, COUNTIF(event_name='purchase') AS enrollments
    FROM `{export_ds}.events_*`
    WHERE {suffix_pred}
    GROUP BY date
    ORDER BY date
    """
    enr = {r['date']: int(r['enrollments'] or 0) for r in query(client, q_enr)}

    # 2) D2P: purchases without a trial signal in prior d2p_look days
    q_d2p = f"""
    WITH purchases AS (
      SELECT user_pseudo_id, PARSE_DATE('%Y%m%d', event_date) AS d, TIMESTAMP_MICROS(event_timestamp) AS pts
      FROM `{export_ds}.events_*`
      WHERE {suffix_pred} AND event_name='purchase'
    ), trials AS (
      SELECT user_pseudo_id, TIMESTAMP_MICROS(event_timestamp) AS tts
      FROM `{export_ds}.events_*` e, UNNEST(event_params) ep
      WHERE (
        _TABLE_SUFFIX BETWEEN FORMAT_DATE('%Y%m%d', DATE_SUB(PARSE_DATE('%Y%m%d','{start}'), INTERVAL {d2p_look} DAY)) AND '{end}'
        OR _TABLE_SUFFIX IN ('{intraday_prev}', '{intraday_end}')
      )
        AND e.event_name='form_submit' AND ep.key='page_location'
        AND REGEXP_CONTAINS(ep.value.string_value, @trial_regex)
    )
    SELECT d AS date, COUNT(1) AS d2p_starts
    FROM purchases p
    LEFT JOIN trials t
      ON t.user_pseudo_id=p.user_pseudo_id
     AND t.tts BETWEEN TIMESTAMP_SUB(p.pts, INTERVAL {d2p_look} DAY) AND p.pts
    WHERE t.user_pseudo_id IS NULL
    GROUP BY date
    ORDER BY date
    """
    d2p = {r['date']: int(r['d2p_starts'] or 0) for r in query_params(client, q_d2p, {'trial_regex': trial_regex})}

    # 3) Post-trial: first purchase within post_win days after trial
    q_pt = f"""
    WITH trials AS (
      SELECT user_pseudo_id, TIMESTAMP_MICROS(event_timestamp) AS tts
      FROM `{export_ds}.events_*` e, UNNEST(event_params) ep
      WHERE {suffix_pred}
        AND e.event_name='form_submit' AND ep.key='page_location'
        AND REGEXP_CONTAINS(ep.value.string_value, @trial_regex)
    ), purchases AS (
      SELECT user_pseudo_id, TIMESTAMP_MICROS(event_timestamp) AS pts
      FROM `{export_ds}.events_*`
      WHERE {suffix_pred} AND event_name='purchase'
    ), first_match AS (
      SELECT t.user_pseudo_id, MIN(p.pts) AS first_purchase
      FROM trials t JOIN purchases p
        ON p.user_pseudo_id=t.user_pseudo_id
       AND p.pts BETWEEN t.tts AND TIMESTAMP_ADD(t.tts, INTERVAL {post_win} DAY)
      GROUP BY t.user_pseudo_id
    )
    SELECT DATE(first_purchase) AS date, COUNT(1) AS post_trial_subs
    FROM first_match
    GROUP BY date
    ORDER BY date
    """
    post = {r['date']: int(r['post_trial_subs'] or 0) for r in query_params(client, q_pt, {'trial_regex': trial_regex})}

    # 4) Mobile subs heuristic
    # Count purchases with cc/plan_code matching regex; device.category filter is often WEB-only in this export
    where_mobile = []
    if cc_re:
        where_mobile.append(f"(ep_cc.key='cc' AND REGEXP_CONTAINS(ep_cc.value.string_value, \"{cc_re}\"))")
    if plan_re:
        where_mobile.append(f"(ep_plan.key='plan_code' AND REGEXP_CONTAINS(ep_plan.value.string_value, \"{plan_re}\"))")
    mobile_pred = ' OR '.join(where_mobile) or 'FALSE'
    q_m = f"""
    WITH p AS (
      SELECT PARSE_DATE('%Y%m%d', e.event_date) AS date
      FROM `{export_ds}.events_*` e
      LEFT JOIN UNNEST(e.event_params) ep_cc ON ep_cc.key='cc'
      LEFT JOIN UNNEST(e.event_params) ep_plan ON ep_plan.key='plan_code'
      WHERE {suffix_pred}
        AND e.event_name='purchase'
        AND ({mobile_pred})
    )
    SELECT date, COUNT(1) AS mobile_subs
    FROM p GROUP BY date ORDER BY date
    """
    # No params used here (predicates already injected); if needed, switch to query_params
    mob = {r['date']: int(r['mobile_subs'] or 0) for r in query(client, q_m)}

    # Merge per date
    all_dates = sorted(set(enr.keys()) | set(d2p.keys()) | set(post.keys()) | set(mob.keys()))
    rows = []
    for d in all_dates:
        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'enrollments': enr.get(d, 0),
            'd2p_starts': d2p.get(d, 0),
            'post_trial_subs': post.get(d, 0),
            'mobile_subs': mob.get(d, 0),
        })

    # Write ga4_derived_daily
    table_id = f"{project}.{dataset}.ga4_derived_daily"
    schema = [
        bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
        bigquery.SchemaField('enrollments', 'INT64'),
        bigquery.SchemaField('d2p_starts', 'INT64'),
        bigquery.SchemaField('post_trial_subs', 'INT64'),
        bigquery.SchemaField('mobile_subs', 'INT64'),
    ]
    client.delete_table(table_id, not_found_ok=True)
    ensure_table(client, table_id, schema)
    job = client.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()

    # Monthly roll-up
    q_month = f"""
    SELECT DATE_TRUNC(date, MONTH) AS month,
           SUM(enrollments) AS enrollments,
           SUM(d2p_starts) AS d2p_starts,
           SUM(post_trial_subs) AS post_trial_subs,
           SUM(mobile_subs) AS mobile_subs
    FROM `{table_id}`
    GROUP BY month ORDER BY month
    """
    monthly = [
        {
            'month': r['month'].strftime('%Y-%m-%d'),
            'enrollments': int(r['enrollments'] or 0),
            'd2p_starts': int(r['d2p_starts'] or 0),
            'post_trial_subs': int(r['post_trial_subs'] or 0),
            'mobile_subs': int(r['mobile_subs'] or 0),
        }
        for r in query(client, q_month, location='us-central1')
    ]
    month_id = f"{project}.{dataset}.ga4_derived_monthly"
    m_schema = [
        bigquery.SchemaField('month', 'DATE', mode='REQUIRED'),
        bigquery.SchemaField('enrollments', 'INT64'),
        bigquery.SchemaField('d2p_starts', 'INT64'),
        bigquery.SchemaField('post_trial_subs', 'INT64'),
        bigquery.SchemaField('mobile_subs', 'INT64'),
    ]
    client.delete_table(month_id, not_found_ok=True)
    ensure_table(client, month_id, m_schema)
    job2 = client.load_table_from_json(monthly, month_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job2.result()


def build_offer_code_daily(client: bigquery.Client, project: str, dataset: str, export_ds: str, cfg: Dict[str, Any]) -> None:
    days_back = int(os.getenv('GA4_DERIVED_DAYS', cfg.get('days_back', 120)))
    start = (date.today() - timedelta(days=days_back)).strftime('%Y%m%d')
    end = date.today().strftime('%Y%m%d')
    q = f"""
    WITH base AS (
      SELECT PARSE_DATE('%Y%m%d', e.event_date) AS date,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='plan_code') AS plan_code,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='cc') AS cc,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='source_cookie') AS source,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='medium_cookie') AS medium,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='landing_page_cookie') AS landing,
             (SELECT ANY_VALUE(ep.value.string_value) FROM UNNEST(e.event_params) ep WHERE ep.key='initial_payment_revenue') AS init_rev
      FROM `{export_ds}.events_*` e
      WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' AND e.event_name='purchase'
    )
    SELECT date,
           plan_code, cc, source, medium,
           SAFE_CAST(REGEXP_EXTRACT(landing, r"https?://([^/]+)") AS STRING) AS host,
           COUNT(1) AS purchases,
           SUM(SAFE_CAST(init_rev AS FLOAT64)) AS initial_payment_revenue
    FROM base
    GROUP BY date, plan_code, cc, source, medium, host
    ORDER BY date
    """
    rows = [dict(r) for r in query(client, q)]
    # Normalize None to None and cast types
    for r in rows:
        r['date'] = r['date'].strftime('%Y-%m-%d')
        r['purchases'] = int(r['purchases'] or 0)
        r['initial_payment_revenue'] = float(r['initial_payment_revenue'] or 0.0)

    table_id = f"{project}.{dataset}.ga4_offer_code_daily"
    schema = [
        bigquery.SchemaField('date', 'DATE', 'REQUIRED'),
        bigquery.SchemaField('plan_code', 'STRING'),
        bigquery.SchemaField('cc', 'STRING'),
        bigquery.SchemaField('source', 'STRING'),
        bigquery.SchemaField('medium', 'STRING'),
        bigquery.SchemaField('host', 'STRING'),
        bigquery.SchemaField('purchases', 'INT64'),
        bigquery.SchemaField('initial_payment_revenue', 'FLOAT64'),
    ]
    client.delete_table(table_id, not_found_ok=True)
    ensure_table(client, table_id, schema)
    job = client.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()


def main():
    mapping_path = os.environ.get('GA4_MAPPING_PATH', 'AELP2/config/ga4_mapping.yaml')
    cfg = load_mapping(mapping_path)
    client, project, dataset, export_ds = get_clients()

    build_daily(client, project, dataset, export_ds, cfg)
    build_offer_code_daily(client, project, dataset, export_ds, cfg)
    print('Built ga4_derived_daily, ga4_derived_monthly, ga4_offer_code_daily')


if __name__ == '__main__':
    main()
