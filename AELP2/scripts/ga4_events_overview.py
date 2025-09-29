#!/usr/bin/env python3
"""
List top GA4 events and counts over a lookback window to help map to pacer metrics.

Requires: GA4_PROPERTY_ID and GA4 auth (service account or OAuth refresh).

Usage:
  export GA4_PROPERTY_ID=properties/XXXX
  python3 AELP2/scripts/ga4_events_overview.py --days 120 --limit 100
"""

import os
import argparse
from datetime import date, timedelta

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
from google.oauth2.credentials import Credentials as OAuthCredentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google.auth.transport.requests import Request as OAuthRequest
import google.auth

SCOPES = ["https://www.googleapis.com/auth/analytics.readonly"]


def get_client():
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
    ap.add_argument('--days', type=int, default=120)
    ap.add_argument('--limit', type=int, default=200)
    args = ap.parse_args()
    prop = os.getenv('GA4_PROPERTY_ID')
    if not prop:
        raise SystemExit('Set GA4_PROPERTY_ID=properties/XXXX')
    client = get_client()
    start = str(date.today() - timedelta(days=args.days))
    end = str(date.today())
    req = RunReportRequest(
        property=prop,
        date_ranges=[DateRange(start_date=start, end_date=end)],
        dimensions=[Dimension(name='eventName')],
        metrics=[Metric(name='eventCount')],
        order_bys=[{"metric": {"metric_name": "eventCount"}, "desc": True}],
        limit=args.limit,
    )
    resp = client.run_report(req)
    for r in resp.rows:
        print(f"{r.dimension_values[0].value}\t{r.metric_values[0].value}")


if __name__ == '__main__':
    main()

