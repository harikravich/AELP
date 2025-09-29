#!/usr/bin/env python3
"""
List top GA4 event names for the configured property so we can pick the
"enrollment" event used in the Daily Enrollments exploration.

Env required:
  - GA4_PROPERTY_ID = properties/<id>

This uses the GA4 Data API. Auth order:
  1) OAuth refresh token (GA4_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN)
  2) Service Account JSON at GOOGLE_APPLICATION_CREDENTIALS
  3) ADC with analytics.readonly scope
"""
import os
import sys
from datetime import date, timedelta

try:
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, RunReportRequest
    from google.oauth2.credentials import Credentials as OAuthCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google.auth.transport.requests import Request as OAuthRequest
    import google.auth
except Exception as e:
    print("google-analytics-data not installed. Install with: python3 -m pip install --user google-analytics-data", file=sys.stderr)
    raise

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

    # ADC
    adc, _ = google.auth.default(scopes=SCOPES)
    return BetaAnalyticsDataClient(credentials=adc)


def main():
    prop = os.getenv("GA4_PROPERTY_ID")
    if not prop:
        print("Set GA4_PROPERTY_ID = properties/<id>", file=sys.stderr)
        sys.exit(2)

    days = int(os.getenv("DAYS", "60"))
    client = get_client()

    req = RunReportRequest(
        property=prop,
        date_ranges=[DateRange(start_date=str(date.today() - timedelta(days=days)), end_date=str(date.today()))],
        dimensions=[Dimension(name="eventName")],
        metrics=[Metric(name="eventCount")],
        limit=200,
        order_bys=[{"desc": True, "metric": {"metric_name": "eventCount"}}],
    )
    resp = client.run_report(req)
    rows = [(r.dimension_values[0].value, int(float(r.metric_values[0].value or 0))) for r in resp.rows]

    # Heuristic ranking for likely enrollment events
    keywords = ("enroll", "signup", "sign_up", "purchase", "start", "apply", "complete", "register")
    likely = [r for r in rows if any(k in r[0].lower() for k in keywords)]

    print("Top events (last {} days):".format(days))
    for name, cnt in rows[:30]:
        print(f"  {name:40s} {cnt}")

    print("\nLikely enrollment candidates:")
    for name, cnt in likely[:20]:
        print(f"  {name:40s} {cnt}")


if __name__ == "__main__":
    main()

