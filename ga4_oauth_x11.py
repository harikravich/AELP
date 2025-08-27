#!/usr/bin/env python3
"""
GA4 OAuth with X11 forwarding - Complete authentication flow
"""

import os
import sys
import pickle
import signal
from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
)
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Configuration
GA_PROPERTY_ID = "308028264"
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_x11_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

# Timeout handler
def timeout_handler(signum, frame):
    print("\n⏱️ Authentication timed out after 120 seconds")
    print("The browser may not have completed the OAuth flow")
    sys.exit(1)

print("\n" + "="*70)
print("GA4 OAUTH WITH X11 FORWARDING")
print("="*70)

# Check X11
if 'DISPLAY' not in os.environ:
    # SSH X11 forwarding usually sets this automatically
    os.environ['DISPLAY'] = ':10.0'  # Common SSH X11 forwarding value
    print(f"⚠️  DISPLAY not set, using default: {os.environ['DISPLAY']}")
else:
    print(f"✅ Using DISPLAY={os.environ['DISPLAY']}")

# OAuth configuration using Google's test client
oauth_config = {
    "installed": {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"]
    }
}

try:
    print("\n🔐 Starting OAuth flow...")
    print("This will open Firefox on your laptop")
    
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2 minute timeout
    
    flow = InstalledAppFlow.from_client_config(
        oauth_config,
        scopes=SCOPES
    )
    
    print("\n📋 Opening browser for authentication...")
    print("Please complete the login in Firefox")
    print("Note: You may see an error about the app not complying with policies")
    print("This is because we're using Google's test OAuth client")
    
    # This will open browser and wait for redirect
    creds = flow.run_local_server(
        port=8080,
        authorization_prompt_message='',
        success_message='Authentication successful! You can close this window.',
        open_browser=True
    )
    
    # Cancel timeout
    signal.alarm(0)
    
    print("\n✅ Got credentials!")
    
    # Save credentials
    with open(TOKEN_FILE, 'wb') as token:
        pickle.dump(creds, token)
    print(f"💾 Saved token to {TOKEN_FILE}")
    
    # Test connection
    print("\n🔍 Testing GA4 access...")
    client = BetaAnalyticsDataClient(credentials=creds)
    
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name="sessions")],
        limit=3
    )
    
    response = client.run_report(request)
    
    print(f"\n✅ SUCCESS! Connected to GA4 property {GA_PROPERTY_ID}")
    print("\nSample data (last 7 days):")
    for row in response.rows:
        channel = row.dimension_values[0].value or "(not set)"
        sessions = row.metric_values[0].value
        print(f"  {channel}: {sessions} sessions")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    if "doesn't comply" in str(e):
        print("\nGoogle is blocking the test OAuth client")
        print("We need to use the service account instead")
    elif "403" in str(e):
        print("\nYour account doesn't have access to this GA4 property")
    else:
        print(f"\nFull error: {e}")
    sys.exit(1)