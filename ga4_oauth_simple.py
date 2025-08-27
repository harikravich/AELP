#!/usr/bin/env python3
"""
GA4 OAuth - Simple manual version
Just generates the URL and waits for code
"""

import pickle
import requests
from pathlib import Path
from urllib.parse import urlencode
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
)
from google.oauth2.credentials import Credentials

# Configuration
GA_PROPERTY_ID = "308028264"
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

# OAuth endpoints
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"

# Using Google's test OAuth client
CLIENT_ID = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
CLIENT_SECRET = "d-FL95Q19q7MQmFpd7hHD0Ty"

print("\n" + "="*70)
print("GA4 OAUTH - MANUAL AUTHENTICATION")
print("="*70)

# Generate OAuth URL
params = {
    'client_id': CLIENT_ID,
    'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob',
    'response_type': 'code',
    'scope': ' '.join(SCOPES),
    'access_type': 'offline',
    'prompt': 'consent'
}

auth_url = f"{AUTH_URI}?{urlencode(params)}"

print("\n📋 INSTRUCTIONS:")
print("="*70)
print("\n1. Open this URL in a browser (on your laptop or anywhere):\n")
print(auth_url)
print("\n2. Sign in with hari@aura.com")
print("\n3. You'll likely see an error about OAuth compliance")
print("\n4. If you get a code, paste it below")
print("\n" + "="*70)

code = input("\nPaste authorization code (or press Enter to skip): ").strip()

if code:
    print("\n🔄 Exchanging code for token...")
    
    token_data = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code',
        'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
    }
    
    try:
        response = requests.post(TOKEN_URI, data=token_data)
        response.raise_for_status()
        tokens = response.json()
        
        # Create credentials
        creds = Credentials(
            token=tokens['access_token'],
            refresh_token=tokens.get('refresh_token'),
            token_uri=TOKEN_URI,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            scopes=SCOPES
        )
        
        # Save token
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)
        
        print("✅ Token saved!")
        
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
        
        print(f"\n✅ SUCCESS! Connected to GA4")
        print("\nSample data:")
        for row in response.rows:
            channel = row.dimension_values[0].value or "(not set)"
            sessions = row.metric_values[0].value
            print(f"  {channel}: {sessions} sessions")
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("\n⚠️  No code provided")
    print("\nThis is expected - Google blocks their test OAuth client")
    print("\nWe need to wait for Jason to add the service account to GA4")
    print("\nService account email: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com")