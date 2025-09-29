#!/usr/bin/env python3
"""
Build GA4 → Pacer metrics daily table in BigQuery using GA4 Data API.

Inputs:
  - GA4 property: env GA4_PROPERTY_ID (e.g., properties/123456789)
  - Auth: service account via GOOGLE_APPLICATION_CREDENTIALS, or OAuth refresh (GA4_OAUTH_REFRESH_TOKEN, GA4_OAUTH_CLIENT_ID, GA4_OAUTH_CLIENT_SECRET)
  - Mapping file: AELP2/config/ga4_pacer_mapping.yaml

Outputs:
  - <project>.<dataset>.ga4_pacer_daily
    date DATE,
    free_trial_starts INT64,
    app_trial_starts INT64,
    d2p_starts INT64,
    post_trial_subscribers INT64,
    mobile_subscribers INT64,
    d2c_total_subscribers INT64

Usage:
  export GOOGLE_CLOUD_PROJECT=...
  export BIGQUERY_TRAINING_DATASET=gaelp_training GA4_PROPERTY_ID=properties/XXXX
  python3 AELP2/scripts/ga4_pacer_to_bq.py --days 400 [--mapping AELP2/config/ga4_pacer_mapping.yaml]
"""

import os
import json
import argparse
from datetime import date, timedelta
from typing import Dict, Any, List, Optional

import yaml
from google.cloud import bigquery
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest, OrderBy
from google.analytics.data_v1beta.types import Filter, FilterExpression, FilterExpressionList
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request as OAuthRequest
import google.auth

SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]


def get_client() -> BetaAnalyticsDataClient:
    rt = os.getenv("GA4_OAUTH_REFRESH_TOKEN")
    cid = os.getenv("GA4_OAUTH_CLIENT_ID")
    cs = os.getenv("GA4_OAUTH_CLIENT_SECRET")
    if rt and cid and cs:
        creds = OAuthCredentials(
            token=None,
            refresh_token=rt,
            client_id=cid,
            client_secret=cs,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=SCOPES,
        )
        creds.refresh(OAuthRequest())
        return BetaAnalyticsDataClient(credentials=creds)
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and os.path.isfile(os.path.expanduser(sa_path)):
        creds = ServiceAccountCredentials.from_service_account_file(os.path.expanduser(sa_path), scopes=SCOPES)
        return BetaAnalyticsDataClient(credentials=creds)
    adc, _ = google.auth.default(scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=adc)


def run_count(
    client: BetaAnalyticsDataClient,
    prop: str,
    ev_name: str,
    days: int,
    device_category: Optional[str] = None,
    contains_item_name: Optional[str] = None,
    contains_page: Optional[List[str]] = None,
    source: Optional[str] = None,
    medium: Optional[str] = None,
) -> Dict[str, int]:
    start = str(date.today() - timedelta(days=days))
    end = str(date.today())
    filters: List[FilterExpression] = [
        FilterExpression(filter=Filter(field_name='eventName', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=ev_name)))
    ]
    if device_category:
        filters.append(FilterExpression(filter=Filter(field_name='deviceCategory', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=device_category))))
    if source and medium:
        filters.append(FilterExpression(filter=Filter(field_name='sessionSourceMedium', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=f"{source} / {medium}"))))
    else:
        if source:
            filters.append(FilterExpression(filter=Filter(field_name='source', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=source))))
        if medium:
            filters.append(FilterExpression(filter=Filter(field_name='medium', string_filter=Filter.StringFilter.MatchType.EXACT, value=medium)))

    # Build dimension filter: supports itemName contains and/or pageLocation contains
    dim_filter: Optional[FilterExpression] = None
    dim_filters: List[FilterExpression] = []
    if contains_item_name:
        dim_filters.append(FilterExpression(filter=Filter(field_name='itemName', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.CONTAINS, value=contains_item_name))))
    if contains_page:
        # OR over the page substrings, wrapped and ANDed with others
        or_list = [FilterExpression(filter=Filter(field_name='pageLocation', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.CONTAINS, value=s))) for s in contains_page]
        if or_list:
            dim_filters.append(FilterExpression(or_group=FilterExpressionList(expressions=or_list)))
    req = RunReportRequest(
        property=prop,
        date_ranges=[DateRange(start_date=start, end_date=end)],
        dimensions=[Dimension(name='date')],
        metrics=[Metric(name='eventCount')],
        dimension_filter=FilterExpression(and_group=FilterExpressionList(expressions=filters + dim_filters)) if (dim_filters or filters) else None,
        limit=200000,
    )
    resp = client.run_report(req)
    out: Dict[str, int] = {}
    for r in resp.rows:
        d = r.dimension_values[0].value
        out[f"{d[0:4]}-{d[4:6]}-{d[6:8]}"] = int(float(r.metric_values[0].value or 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', default='AELP2/config/ga4_pacer_mapping.yaml')
    ap.add_argument('--days', type=int, default=400)
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    prop = os.getenv('GA4_PROPERTY_ID')
    if not (project and dataset and prop):
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, and GA4_PROPERTY_ID')

    with open(args.mapping, 'r') as f:
        mapping = yaml.safe_load(f)

    client = get_client()
    # Collect per metric daily counts
    metric_to_series: Dict[str, Dict[str,int]] = {}
    for metric, spec in mapping.items():
        ev = spec.get('event_name')
        device_category = spec.get('device_category')
        if not ev:
            print(f"[warn] {metric}: no event_name configured; skipping")
            continue
        series = run_count(client, prop, ev, args.days, device_category=device_category)
        metric_to_series[metric] = series

    # Build union of dates
    all_dates = sorted({d for s in metric_to_series.values() for d in s.keys()})
    rows: List[Dict[str, Any]] = []
    for d in all_dates:
        row = {'date': d}
        for m in mapping.keys():
            row[m] = int(metric_to_series.get(m,{}).get(d, 0))
        rows.append(row)

    # Write to BigQuery (prefer GCE ADC; avoid reusing GA4 SA JSON for BQ)
    os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    bq = bigquery.Client(project=project)
    table_id = f"{project}.{dataset}.ga4_pacer_daily"
    schema = [bigquery.SchemaField('date','DATE','REQUIRED')] + [bigquery.SchemaField(k,'INT64','NULLABLE') for k in mapping.keys()]
    try:
        bq.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field='date')
    bq.create_table(table)
    job = bq.load_table_from_json(rows, table_id, job_config=bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE'))
    job.result()
    print(f"Wrote {len(rows)} rows to {table_id} using mapping {args.mapping}")


if __name__ == '__main__':
    main()
