#!/usr/bin/env python3
"""
GA4 OAuth Headless - Works without browser
"""

import pickle
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Configuration
GA_PROPERTY_ID = "308028264"
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("GA4 OAUTH - MANUAL URL METHOD")
print("="*70)

# OAuth config
oauth_config = {
    "installed": {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8888/"]
    }
}

flow = InstalledAppFlow.from_client_config(oauth_config, scopes=SCOPES, redirect_uri="http://localhost:8888/")
auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')

print("\n1. Open this URL on YOUR LAPTOP in Chrome/Firefox:\n")
print(auth_url)
print("\n2. Sign in with hari@aura.com (handles OKTA)")
print("\n3. After auth, you'll be redirected to localhost:8888")
print("   The page will error (expected) but copy the FULL URL")
print("\n4. Paste the ENTIRE URL below (with code= parameter)")

redirect_url = input("\nPaste URL: ").strip()

if redirect_url:
    parsed = urlparse(redirect_url)
    params = parse_qs(parsed.query)
    if 'code' in params:
        code = params['code'][0]
        print("\n✅ Got code, exchanging for token...")
        
        flow.fetch_token(code=code)
        creds = flow.credentials
        
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)
        print("✅ Token saved!")
        
        # Test
        client = BetaAnalyticsDataClient(credentials=creds)
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
            dimensions=[Dimension(name="sessionDefaultChannelGroup")],
            metrics=[Metric(name="sessions")],
            limit=3
        )
        
        response = client.run_report(request)
        print(f"\n✅ SUCCESS! Connected to GA4")
        for row in response.rows:
            print(f"  {row.dimension_values[0].value}: {row.metric_values[0].value} sessions")
    else:
        print("❌ No code in URL")
else:
    print("❌ No URL provided")
