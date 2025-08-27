#!/usr/bin/env python3
"""
Explore GA4 Data for GAELP Calibration
"""

from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import *
from google.oauth2 import service_account

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

print("\nUSER JOURNEY ANALYSIS:")
print("-" * 70)

# Get conversion funnel
request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="sessionDefaultChannelGroup"),
        Dimension(name="deviceCategory"),
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="averageSessionDuration"),
    ],
    limit=20
)

response = client.run_report(request)
for row in response.rows:
    channel = row.dimension_values[0].value or "(not set)"
    device = row.dimension_values[1].value
    sessions = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    duration = float(row.metric_values[2].value)
    
    if sessions > 1000:
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        print(f"{channel:20} | {device:7} | CVR: {cvr:.2f}% | {duration:.0f}s")
