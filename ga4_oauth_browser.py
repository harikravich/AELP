#!/usr/bin/env python3
"""
GA4 OAuth with Browser via X11 - Uses localhost redirect (not OOB)
This works with X11 forwarding and browser
"""

import os
import pickle
import webbrowser
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
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_browser_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("GA4 OAUTH WITH BROWSER (X11) - Using localhost redirect")
print("="*70)

# OAuth configuration - USING LOCALHOST, NOT OOB!
oauth_config = {
    "installed": {
        "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
        "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:8080/", "http://localhost"]  # NOT urn:ietf:wg:oauth:2.0:oob
    }
}

# Check for saved token first
if TOKEN_FILE.exists():
    try:
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
        print("✅ Found saved token, testing connection...")
        
        client = BetaAnalyticsDataClient(credentials=creds)
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="yesterday", end_date="today")],
            dimensions=[Dimension(name="sessionDefaultChannelGroup")],
            metrics=[Metric(name="sessions")],
            limit=1
        )
        response = client.run_report(request)
        print("✅ Token still valid! GA4 access confirmed.")
        exit(0)
    except Exception as e:
        print(f"⚠️ Saved token expired or invalid: {e}")
        print("Proceeding with new authentication...")

try:
    print("\n🔐 Starting OAuth flow with localhost redirect...")
    print("This uses localhost callback, not OOB (which Google blocked)")
    
    flow = InstalledAppFlow.from_client_config(
        oauth_config,
        scopes=SCOPES
    )
    
    # Check if we can open browser
    if 'DISPLAY' in os.environ:
        print(f"✅ X11 detected (DISPLAY={os.environ['DISPLAY']})")
        print("\n📋 Starting authentication server on localhost:8080...")
        print("The browser should open automatically via X11")
        print("\nNOTE: Google may still block the test OAuth client")
        print("But at least we're not using the blocked OOB flow")
        
        # This will:
        # 1. Start a local server on port 8080
        # 2. Open browser via X11 to the OAuth URL
        # 3. Redirect back to localhost:8080 after auth
        # 4. Capture the authorization code automatically
        creds = flow.run_local_server(
            port=8080,
            authorization_prompt_message='Browser opening via X11...\n',
            success_message='Authentication successful! You can close this browser window.',
            open_browser=True  # Will use X11 to open browser
        )
        
        print("\n✅ Got credentials via browser!")
        
    else:
        print("❌ No X11 display found")
        print("\nFor manual authentication, open this URL in any browser:")
        auth_url, _ = flow.authorization_url(
            access_type='offline',
            prompt='consent'
        )
        print(f"\n{auth_url}")
        print("\nAfter authorizing, you'll be redirected to localhost:8080")
        print("Copy the FULL URL from the browser and paste here:")
        
        # Manual flow - user pastes the redirect URL
        redirect_response = input("\nPaste the full redirect URL: ").strip()
        flow.fetch_token(authorization_response=redirect_response)
        creds = flow.credentials
    
    # Save credentials
    with open(TOKEN_FILE, 'wb') as f:
        pickle.dump(creds, f)
    print(f"💾 Saved token to {TOKEN_FILE}")
    
    # Test connection
    print("\n🔍 Testing GA4 access...")
    client = BetaAnalyticsDataClient(credentials=creds)
    
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[Metric(name="sessions"), Metric(name="conversions")],
        limit=5
    )
    
    response = client.run_report(request)
    
    print(f"\n✅ SUCCESS! Connected to GA4 property {GA_PROPERTY_ID}")
    print("\nSample data (last 7 days):")
    for row in response.rows:
        channel = row.dimension_values[0].value or "(not set)"
        sessions = row.metric_values[0].value
        conversions = row.metric_values[1].value
        print(f"  {channel}: {sessions} sessions, {conversions} conversions")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    
    if "doesn't comply" in str(e) or "invalid_client" in str(e):
        print("\n⚠️ Google is blocking the test OAuth client")
        print("\nThe real solution is to use the service account:")
        print("Service account: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com")
        print("\nJason needs to add this service account to GA4 with Viewer access")
        
    elif "403" in str(e):
        print("\n⚠️ Your account doesn't have access to this GA4 property")
        print("Verify that hari@aura.com has access to property ID: 308028264")
        
    else:
        print(f"\nFull error details: {e}")