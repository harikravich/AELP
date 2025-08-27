#!/usr/bin/env python3
"""
GA4 OAuth - Manual authorization from your laptop
Get the code from your laptop browser, use it here on GCP
"""

import os
import json
import pickle
import requests
from pathlib import Path
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
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_token_manual.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

class GA4ManualOAuth:
    def __init__(self):
        self.creds = None
        self.client = None
        self.property = f"properties/{GA_PROPERTY_ID}"
        
        # Using Google's test OAuth client
        self.client_id = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
        self.client_secret = "d-FL95Q19q7MQmFpd7hHD0Ty"
        
    def get_auth_url(self):
        """Generate OAuth URL to open on your laptop"""
        from urllib.parse import urlencode
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob',  # This gives you a code to copy
            'response_type': 'code',
            'scope': ' '.join(SCOPES),
            'access_type': 'offline',
            'prompt': 'consent'
        }
        
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
        return auth_url
    
    def exchange_code(self, auth_code):
        """Exchange authorization code for tokens"""
        token_url = "https://oauth2.googleapis.com/token"
        
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': auth_code,
            'grant_type': 'authorization_code',
            'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
        }
        
        try:
            response = requests.post(token_url, data=token_data)
            response.raise_for_status()
            tokens = response.json()
            
            # Create credentials
            self.creds = Credentials(
                token=tokens['access_token'],
                refresh_token=tokens.get('refresh_token'),
                token_uri=token_url,
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=SCOPES
            )
            
            # Save for future use
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(self.creds, token)
            
            print("✅ Authentication successful! Token saved.")
            self.client = BetaAnalyticsDataClient(credentials=self.creds)
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def test_connection(self):
        """Test GA4 access"""
        if not self.client:
            print("Not authenticated")
            return False
            
        try:
            request = RunReportRequest(
                property=self.property,
                date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
                dimensions=[Dimension(name="sessionDefaultChannelGroup")],
                metrics=[Metric(name="sessions"), Metric(name="conversions")],
                limit=5
            )
            
            response = self.client.run_report(request)
            
            print(f"\n✅ CONNECTED TO GA4!")
            print(f"Property: {GA_PROPERTY_ID}")
            print(f"\nLast 7 days:")
            
            for row in response.rows:
                channel = row.dimension_values[0].value or "(not set)"
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                print(f"  {channel}: {sessions:,} sessions, {conversions} conversions")
            
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            if "403" in str(e):
                print("\nYour account doesn't have access to this GA4 property")
            return False

if __name__ == "__main__":
    oauth = GA4ManualOAuth()
    
    # Check for saved token first
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, 'rb') as f:
                oauth.creds = pickle.load(f)
            oauth.client = BetaAnalyticsDataClient(credentials=oauth.creds)
            print("✅ Using saved credentials")
            oauth.test_connection()
            exit(0)
        except:
            pass
    
    print("\n" + "="*70)
    print("GA4 OAUTH - MANUAL AUTHORIZATION")
    print("="*70)
    
    auth_url = oauth.get_auth_url()
    
    print("\n📋 INSTRUCTIONS:")
    print("="*70)
    print("\n1. ON YOUR LAPTOP, open this URL in Chrome/Firefox:\n")
    print(auth_url)
    print("\n2. Sign in with hari@aura.com")
    print("   - It will handle OKTA SSO properly in your browser")
    print("\n3. Grant access to Google Analytics")
    print("\n4. You'll see a code like: 4/0AQlEd8...")
    print("\n5. Copy that ENTIRE code")
    print("\n" + "="*70)
    
    auth_code = input("\nPaste the authorization code here: ").strip()
    
    if auth_code:
        if oauth.exchange_code(auth_code):
            oauth.test_connection()
    else:
        print("No code provided")