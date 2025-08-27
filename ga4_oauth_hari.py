#!/usr/bin/env python3
"""
GA4 OAuth with hari@aura.com - Direct User Authentication
Create your own OAuth client for proper authentication
"""

import os
import json
import pickle
import webbrowser
from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
)
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Configuration
GA_PROPERTY_ID = "308028264"
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_hari_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\n" + "="*70)
print("GA4 OAUTH SETUP - Using hari@aura.com directly")
print("="*70)

print("""
To make this work, you need to create YOUR OWN OAuth client:

1. Go to: https://console.cloud.google.com/
   
2. Select the project: centering-line-469716-r7
   (or create a new project if you don't have access)

3. Go to "APIs & Services" → "Credentials"

4. Click "+ CREATE CREDENTIALS" → "OAuth client ID"

5. If prompted, configure consent screen:
   - Internal (if possible) or External
   - App name: GAELP GA4 Reader
   - Your email: hari@aura.com
   
6. For OAuth client:
   - Application type: Desktop app
   - Name: GAELP Desktop Client
   
7. Click CREATE and download the JSON file

8. Save it as: ~/.config/gaelp/oauth_client.json

Once you have the client JSON, this script will:
- Use YOUR OAuth client (not Google's test one)
- Authenticate with hari@aura.com
- Access GA4 with your permissions
- Save tokens for future use

This is the PROPER way - using your own OAuth client with your own credentials.
NO SHORTCUTS, NO SIMPLIFICATIONS.
""")

CLIENT_FILE = Path.home() / '.config' / 'gaelp' / 'oauth_client.json'

if not CLIENT_FILE.exists():
    print(f"\n❌ OAuth client file not found: {CLIENT_FILE}")
    print("\nFollow the steps above to create and download your OAuth client.")
    print("Then save it to the location shown above.")
    exit(1)

class GA4DirectOAuth:
    def __init__(self):
        self.creds = None
        self.client = None
        self.property = f"properties/{GA_PROPERTY_ID}"
        
    def authenticate(self):
        """Authenticate using YOUR OAuth client with hari@aura.com"""
        
        # Check for saved credentials
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE, 'rb') as token:
                self.creds = pickle.load(token)
                print("✅ Found saved credentials for hari@aura.com")
        
        # Refresh or get new credentials
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                print("🔄 Refreshing token...")
                self.creds.refresh(Request())
            else:
                print("\n🔐 Authenticating with hari@aura.com...")
                
                # Load YOUR OAuth client configuration
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(CLIENT_FILE),
                    scopes=SCOPES
                )
                
                # Try to open browser automatically
                try:
                    self.creds = flow.run_local_server(
                        port=0,
                        authorization_prompt_message='Opening browser to authenticate...',
                        success_message='Authentication successful! You can close this window.',
                        open_browser=True
                    )
                except:
                    # Fallback to manual URL
                    print("\n📋 Manual authentication required:")
                    auth_url, _ = flow.authorization_url(
                        access_type='offline',
                        include_granted_scopes='true'
                    )
                    print(f"\n1. Open this URL:\n{auth_url}")
                    print("\n2. Sign in with hari@aura.com")
                    print("\n3. Copy the authorization code")
                    
                    code = input("\nPaste code here: ").strip()
                    flow.fetch_token(code=code)
                    self.creds = flow.credentials
                
                # Save credentials
                with open(TOKEN_FILE, 'wb') as token:
                    pickle.dump(self.creds, token)
                print("\n✅ Authenticated and saved token for hari@aura.com")
        
        # Create client
        self.client = BetaAnalyticsDataClient(credentials=self.creds)
        return True
    
    def test_connection(self):
        """Test GA4 access with hari@aura.com credentials"""
        print("\n" + "="*60)
        print("TESTING GA4 CONNECTION")
        print("="*60)
        
        try:
            request = RunReportRequest(
                property=self.property,
                date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
                dimensions=[Dimension(name="sessionDefaultChannelGroup")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="conversions"),
                ],
                limit=5
            )
            
            response = self.client.run_report(request)
            
            print(f"\n✅ SUCCESS! Connected as hari@aura.com")
            print(f"Property ID: {GA_PROPERTY_ID}")
            print(f"\nSample data (last 7 days):")
            
            for row in response.rows:
                channel = row.dimension_values[0].value or "(not set)"
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                print(f"  {channel}: {sessions:,} sessions, {conversions} conversions")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if "403" in str(e):
                print("\nhari@aura.com doesn't have access to this GA4 property")
                print("Ask Jason to verify your access level")
            return False

if __name__ == "__main__":
    # Check if client file exists
    if CLIENT_FILE.exists():
        print(f"\n✅ Found OAuth client file")
        
        connector = GA4DirectOAuth()
        if connector.authenticate():
            connector.test_connection()
    else:
        print(f"\n⚠️  You need to create your OAuth client first")
        print("Follow the instructions above")