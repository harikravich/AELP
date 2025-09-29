#!/usr/bin/env python3
"""
Write GA4 event counts per day for a chosen event into BigQuery
table <project>.<dataset>.ga4_enrollments_daily so reconciliation can
use exact GA4 "Daily Enrollments" even without GA4 export.

Env:
  - GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, GA4_PROPERTY_ID
  - Auth as in ga4_find_enrollment_event.py (OAuth refresh token or SA JSON)

Usage:
  python3 AELP2/scripts/ga4_event_to_bq_enrollments.py --event purchase --days 180
"""
import os
import argparse
from datetime import date, timedelta

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--event', required=True)
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--channel-group', help='sessionDefaultChannelGroup to include (e.g., Paid Search)')
    ap.add_argument('--source', help='source to include (e.g., google)')
    ap.add_argument('--medium', help='medium to include (e.g., cpc)')
    ap.add_argument('--table-suffix', help='write to ga4_enrollments_<suffix>_daily instead of ga4_enrollments_daily')
    ap.add_argument('--scope', choices=['session','event'], default='session', help='traffic scope for source/medium filter (default: session)')
    args = ap.parse_args()

    project = os.getenv('GOOGLE_CLOUD_PROJECT')
    dataset = os.getenv('BIGQUERY_TRAINING_DATASET')
    prop = os.getenv('GA4_PROPERTY_ID')
    if not (project and dataset and prop):
        raise SystemExit('Set GOOGLE_CLOUD_PROJECT, BIGQUERY_TRAINING_DATASET, GA4_PROPERTY_ID')

    start = str(date.today() - timedelta(days=args.days))
    end = str(date.today())

    client = get_client()
    # Build dimension filter: eventName AND optional channel/source/medium
    filters = [FilterExpression(filter=Filter(field_name='eventName', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=args.event)))]
    if args.channel_group:
        filters.append(FilterExpression(filter=Filter(field_name='sessionDefaultChannelGroup', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=args.channel_group))))
    # Prefer event-scoped sourceMedium when both are provided to match GA4 Exploration "source / medium"
    if args.source and args.medium:
        field = 'sessionSourceMedium' if args.scope == 'session' else 'sourceMedium'
        filters.append(FilterExpression(filter=Filter(field_name=field, string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=f"{args.source} / {args.medium}"))))
    else:
        if args.source:
            filters.append(FilterExpression(filter=Filter(field_name='source', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=args.source))))
        if args.medium:
            filters.append(FilterExpression(filter=Filter(field_name='medium', string_filter=Filter.StringFilter(match_type=Filter.StringFilter.MatchType.EXACT, value=args.medium))))

    req = RunReportRequest(
        property=prop,
        date_ranges=[DateRange(start_date=start, end_date=end)],
        dimensions=[Dimension(name='date')],
        metrics=[Metric(name='eventCount')],
        dimension_filter=FilterExpression(and_group=FilterExpressionList(expressions=filters)),
        limit=100000,
    )
    resp = client.run_report(req)
    rows = []
    for r in resp.rows:
        d = r.dimension_values[0].value
        cnt = int(float(r.metric_values[0].value or 0))
        rows.append({'date': f'{d[0:4]}-{d[4:6]}-{d[6:8]}', 'enrollments': cnt})

    # Use GCE ADC for BigQuery writes even if a GA4 service account JSON is set
    os.environ.pop('GOOGLE_APPLICATION_CREDENTIALS', None)
    bq = bigquery.Client(project=project)
    suffix = args.table_suffix.strip().lower() if args.table_suffix else ''
    table_name = f"ga4_enrollments_{suffix}_daily" if suffix else 'ga4_enrollments_daily'
    table_id = f"{project}.{dataset}.{table_name}"
    # If a view exists, drop it; create a table instead
    try:
        bq.delete_table(table_id, not_found_ok=True)
    except Exception:
        pass
    schema = [
        bigquery.SchemaField('date', 'DATE', mode='REQUIRED'),
        bigquery.SchemaField('enrollments', 'INT64', mode='NULLABLE'),
    ]
    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field='date')
    bq.create_table(table)
    if rows:
        bq.insert_rows_json(table_id, rows)
    print(f"Wrote {len(rows)} rows to {table_id} for event '{args.event}' ({start}..{end})")


if __name__ == '__main__':
    main()
