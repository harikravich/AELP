#!/usr/bin/env python3
"""
GA4 OAuth with YOUR OAuth Client (not Google's test client)
This handles OKTA SSO properly
"""

import json
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
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_hari_token.pickle'
CLIENT_SECRET_FILE = Path.home() / '.config' / 'gaelp' / 'oauth_client_secret.json'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("GA4 OAUTH WITH YOUR CLIENT - Handles OKTA SSO")
print("="*70)

# Check if client secret exists
if not CLIENT_SECRET_FILE.exists():
    print(f"\n❌ OAuth client file not found: {CLIENT_SECRET_FILE}")
    print("\nYou need to create your own OAuth client:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create OAuth 2.0 credentials (Web application type)")
    print("3. Add http://localhost:8888/ to redirect URIs")
    print("4. Download and save to the path above")
    print("\nSee CREATE_OAUTH_CLIENT.md for detailed steps")
    exit(1)

# Check for existing token
if TOKEN_FILE.exists():
    try:
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
        print("✅ Found saved token, testing...")
        
        # Test the token
        client = BetaAnalyticsDataClient(credentials=creds)
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="yesterday", end_date="today")],
            dimensions=[Dimension(name="date")],
            metrics=[Metric(name="sessions")],
            limit=1
        )
        response = client.run_report(request)
        print("✅ Token valid! Already connected to GA4.")
        exit(0)
    except Exception as e:
        print(f"⚠️ Saved token invalid: {e}")
        print("Getting new token...")

# Load YOUR OAuth client configuration
with open(CLIENT_SECRET_FILE, 'r') as f:
    client_config = json.load(f)

# Determine the type of client config
if 'installed' in client_config:
    config_type = 'installed'
elif 'web' in client_config:
    config_type = 'web'
else:
    print("❌ Invalid client configuration file")
    exit(1)

# Create flow with YOUR client
flow = InstalledAppFlow.from_client_secrets_file(
    str(CLIENT_SECRET_FILE),
    scopes=SCOPES,
    redirect_uri="http://localhost:8888/"
)

# Get authorization URL
auth_url, _ = flow.authorization_url(
    access_type='offline',
    prompt='consent'
)

print("\n📋 AUTHENTICATION STEPS:")
print("="*70)
print("\n1. Open this URL on YOUR LAPTOP browser:\n")
print(auth_url)
print("\n2. You'll see Google login")
print("   - Enter: hari@aura.com")
print("   - Google will redirect to OKTA")
print("\n3. Complete OKTA login")
print("   - Use your OKTA credentials")
print("   - Complete any 2FA if required")
print("\n4. After OKTA, you'll return to Google")
print("   - Grant access to Google Analytics")
print("\n5. You'll be redirected to:")
print("   http://localhost:8888/?code=...")
print("   (This will show an error - that's OK!)")
print("\n6. Copy the ENTIRE URL from your browser")
print("   (It should contain 'code=' parameter)")
print("\n" + "="*70)

# Get the redirect URL from user
redirect_url = input("\nPaste the complete redirect URL here: ").strip()

if not redirect_url:
    print("❌ No URL provided")
    exit(1)

# Parse the authorization code
try:
    parsed = urlparse(redirect_url)
    params = parse_qs(parsed.query)
    
    if 'code' not in params:
        print("❌ No authorization code in URL")
        print("Make sure you copied the complete URL including ?code=...")
        exit(1)
    
    auth_code = params['code'][0]
    print(f"\n✅ Got authorization code!")
    
    # Exchange code for token
    print("🔄 Exchanging code for access token...")
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    
    # Save token
    with open(TOKEN_FILE, 'wb') as f:
        pickle.dump(creds, f)
    print(f"💾 Token saved to {TOKEN_FILE}")
    
    # Test GA4 connection
    print("\n🔍 Testing GA4 connection...")
    client = BetaAnalyticsDataClient(credentials=creds)
    
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalUsers"),
            Metric(name="conversions")
        ],
        limit=10
    )
    
    response = client.run_report(request)
    
    print(f"\n✅ SUCCESS! Connected to GA4 Property: {GA_PROPERTY_ID}")
    print("\n📊 Sample Data (Last 30 Days):")
    print("-" * 60)
    
    total_sessions = 0
    total_users = 0
    total_conversions = 0
    
    for row in response.rows:
        channel = row.dimension_values[0].value or "(not set)"
        sessions = int(row.metric_values[0].value)
        users = int(row.metric_values[1].value)
        conversions = int(row.metric_values[2].value)
        
        total_sessions += sessions
        total_users += users
        total_conversions += conversions
        
        print(f"{channel:20} | Sessions: {sessions:,} | Users: {users:,} | Conversions: {conversions}")
    
    print("-" * 60)
    print(f"{'TOTAL':20} | Sessions: {total_sessions:,} | Users: {total_users:,} | Conversions: {total_conversions}")
    
    print("\n✅ GA4 integration complete!")
    print("You can now use GA4 data for GAELP calibration")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    
    if "invalid_grant" in str(e):
        print("\nThe authorization code may have expired.")
        print("Try the process again - codes expire quickly.")
    elif "403" in str(e):
        print("\nYour account doesn't have access to this GA4 property.")
        print("Ask Jason to verify access for hari@aura.com")
    else:
        print(f"\nFull error: {e}")