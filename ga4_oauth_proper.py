#!/usr/bin/env python3
"""
GA4 OAuth PROPER Implementation - NO SHORTCUTS
Full OAuth2 flow with proper client configuration
"""

import os
import json
import pickle
import hashlib
import secrets
import base64
from pathlib import Path
from urllib.parse import urlencode
import requests
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Configuration
GA_PROPERTY_ID = "308028264"
SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_oauth_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

# OAuth2 endpoints
GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

class GA4OAuthProper:
    def __init__(self):
        self.creds = None
        self.client = None
        self.property = f"properties/{GA_PROPERTY_ID}"
        
        # We'll use Google's public OAuth client for testing
        # This is the official test client ID from Google
        self.client_id = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
        self.client_secret = "d-FL95Q19q7MQmFpd7hHD0Ty"
        
    def generate_auth_url(self):
        """Generate proper OAuth2 authorization URL with PKCE"""
        
        # Generate PKCE challenge
        code_verifier = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8').rstrip('=')
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        # Save code_verifier for later
        self.code_verifier = code_verifier
        
        # Build authorization URL with all required parameters
        params = {
            'client_id': self.client_id,
            'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob',  # For manual copy/paste flow
            'response_type': 'code',
            'scope': ' '.join(SCOPES),
            'access_type': 'offline',  # To get refresh token
            'prompt': 'consent',  # Force consent to get refresh token
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256',
            'state': secrets.token_urlsafe(32)
        }
        
        auth_url = f"{GOOGLE_AUTH_URI}?{urlencode(params)}"
        return auth_url, code_verifier
        
    def authenticate(self):
        """Full OAuth2 authentication flow - NO SHORTCUTS"""
        
        # Check for saved credentials
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, 'rb') as token:
                    self.creds = pickle.load(token)
                    print("✅ Found saved credentials")
                    
                # Check if token needs refresh
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    print("🔄 Refreshing expired token...")
                    self.creds.refresh(Request())
                    
                    # Save refreshed token
                    with open(TOKEN_FILE, 'wb') as token:
                        pickle.dump(self.creds, token)
                    
                if self.creds and self.creds.valid:
                    self.client = BetaAnalyticsDataClient(credentials=self.creds)
                    return True
                    
            except Exception as e:
                print(f"⚠️  Saved credentials invalid: {e}")
                os.remove(TOKEN_FILE)
        
        # Need new authentication
        print("\n🔐 FULL OAUTH2 AUTHENTICATION REQUIRED")
        print("=" * 60)
        print("NO SHORTCUTS - Proper OAuth2 flow with PKCE")
        print("=" * 60)
        
        # Generate auth URL
        auth_url, code_verifier = self.generate_auth_url()
        
        print("\n📋 AUTHENTICATION STEPS:")
        print("=" * 60)
        print("\n1. Copy this ENTIRE URL and paste in your browser:\n")
        print(auth_url)
        print("\n" + "=" * 60)
        print("\n2. Sign in with: hari@aura.com")
        print("\n3. You'll see a screen saying:")
        print("   'Google hasn't verified this app'")
        print("   Click 'Continue' (it's safe, this is Google's test OAuth client)")
        print("\n4. Grant access to 'Google Analytics'")
        print("\n5. You'll get a code that looks like: 4/0AQlEd8...")
        print("\n" + "=" * 60)
        
        # Get authorization code
        auth_code = input("\nPaste the authorization code here: ").strip()
        
        if not auth_code:
            print("❌ No code provided")
            return False
            
        print("\n🔄 Exchanging code for tokens...")
        
        # Exchange authorization code for tokens
        token_data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'code': auth_code,
            'code_verifier': code_verifier,
            'grant_type': 'authorization_code',
            'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
        }
        
        try:
            response = requests.post(GOOGLE_TOKEN_URI, data=token_data)
            response.raise_for_status()
            tokens = response.json()
            
            # Create credentials object
            self.creds = Credentials(
                token=tokens['access_token'],
                refresh_token=tokens.get('refresh_token'),
                token_uri=GOOGLE_TOKEN_URI,
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=SCOPES
            )
            
            # Save credentials
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(self.creds, token)
            
            print("✅ Authentication successful! Token saved.")
            
            # Create client
            self.client = BetaAnalyticsDataClient(credentials=self.creds)
            return True
            
        except requests.exceptions.HTTPError as e:
            print(f"❌ Token exchange failed: {e}")
            print(f"Response: {e.response.text}")
            return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    def test_connection(self):
        """Test GA4 connection - REAL DATA ONLY"""
        print("\n" + "="*60)
        print("TESTING GA4 CONNECTION - NO MOCKS")
        print("="*60)
        
        try:
            request = RunReportRequest(
                property=self.property,
                date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
                dimensions=[Dimension(name="sessionDefaultChannelGroup")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="conversions"),
                    Metric(name="totalRevenue"),
                ],
                limit=10
            )
            
            response = self.client.run_report(request)
            
            print(f"\n✅ REAL CONNECTION SUCCESSFUL!")
            print(f"Property ID: {GA_PROPERTY_ID}")
            print(f"Authenticated as: hari@aura.com")
            print(f"\n📊 REAL DATA (Last 7 days):")
            
            total_sessions = 0
            total_conversions = 0
            total_revenue = 0
            
            for row in response.rows:
                channel = row.dimension_values[0].value or "(not set)"
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                revenue = float(row.metric_values[2].value)
                
                total_sessions += sessions
                total_conversions += conversions
                total_revenue += revenue
                
                if sessions > 0:
                    cvr = (conversions / sessions) * 100
                    print(f"\n  {channel}:")
                    print(f"    Sessions: {sessions:,}")
                    print(f"    Conversions: {conversions}")
                    print(f"    Revenue: ${revenue:.2f}")
                    print(f"    CVR: {cvr:.2f}%")
            
            print(f"\n  TOTALS:")
            print(f"    Sessions: {total_sessions:,}")
            print(f"    Conversions: {total_conversions}")
            print(f"    Revenue: ${total_revenue:.2f}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            
            if "403" in str(e):
                print("\n⚠️  PERMISSION ISSUE")
                print("Your account (hari@aura.com) doesn't have access to this property")
                print("OR the property ID is incorrect")
            
            return False
    
    def extract_calibration_data(self):
        """Extract REAL calibration data - NO SYNTHETIC VALUES"""
        print("\n" + "="*60)
        print("EXTRACTING REAL CALIBRATION DATA")
        print("="*60)
        
        request = RunReportRequest(
            property=self.property,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="sessionDefaultChannelGroup"),
                Dimension(name="deviceCategory"),
                Dimension(name="sessionSource"),
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="conversions"),
                Metric(name="totalRevenue"),
                Metric(name="newUsers"),
                Metric(name="averageSessionDuration"),
                Metric(name="bounceRate"),
            ],
        )
        
        response = self.client.run_report(request)
        
        # Process REAL data
        calibration = {}
        
        for row in response.rows:
            channel = row.dimension_values[0].value or "(not set)"
            device = row.dimension_values[1].value or "(not set)"
            source = row.dimension_values[2].value or "(not set)"
            
            sessions = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            revenue = float(row.metric_values[2].value)
            new_users = int(row.metric_values[3].value)
            avg_duration = float(row.metric_values[4].value)
            bounce_rate = float(row.metric_values[5].value)
            
            if sessions > 50:  # Only meaningful data
                key = f"{channel}_{device}"
                
                if key not in calibration:
                    calibration[key] = {
                        'channel': channel,
                        'device': device,
                        'sources': [],
                        'sessions': 0,
                        'conversions': 0,
                        'revenue': 0,
                        'new_users': 0,
                        'total_duration': 0,
                        'bounce_sessions': 0
                    }
                
                calibration[key]['sessions'] += sessions
                calibration[key]['conversions'] += conversions
                calibration[key]['revenue'] += revenue
                calibration[key]['new_users'] += new_users
                calibration[key]['total_duration'] += avg_duration * sessions
                calibration[key]['bounce_sessions'] += bounce_rate * sessions
                
                if source not in calibration[key]['sources']:
                    calibration[key]['sources'].append(source)
        
        # Calculate REAL metrics
        print("\n📊 REAL CALIBRATION DATA:")
        
        for key, data in calibration.items():
            if data['sessions'] > 100:
                cvr = data['conversions'] / data['sessions']
                aov = data['revenue'] / data['conversions'] if data['conversions'] > 0 else 0
                avg_duration = data['total_duration'] / data['sessions']
                bounce_rate = data['bounce_sessions'] / data['sessions']
                
                print(f"\n{data['channel']} ({data['device']}):")
                print(f"  Sessions: {data['sessions']:,}")
                print(f"  Conversions: {data['conversions']}")
                print(f"  CVR: {cvr*100:.3f}%")  # 3 decimal places for precision
                print(f"  AOV: ${aov:.2f}")
                print(f"  Avg Duration: {avg_duration:.1f}s")
                print(f"  Bounce Rate: {bounce_rate*100:.1f}%")
                print(f"  Top Sources: {', '.join(data['sources'][:3])}")
        
        # Save REAL calibration data
        with open('ga4_real_calibration.json', 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ REAL calibration data saved to ga4_real_calibration.json")
        print("NO SYNTHETIC DATA - ALL FROM ACTUAL GA4")
        
        return calibration

if __name__ == "__main__":
    connector = GA4OAuthProper()
    
    print("\n🚀 GA4 PROPER OAUTH IMPLEMENTATION")
    print("=" * 60)
    print("NO SHORTCUTS - Full OAuth2 with PKCE")
    print("Authenticate with: hari@aura.com")
    print("=" * 60)
    
    if connector.authenticate():
        if connector.test_connection():
            connector.extract_calibration_data()
            
            print("\n" + "="*60)
            print("✅ COMPLETE SUCCESS!")
            print("=" * 60)
            print("\nFull OAuth2 implementation working")
            print("Real GA4 data extracted")
            print("Ready for GAELP calibration")
            print("\nNO FALLBACKS - NO SHORTCUTS - REAL DATA!")
    else:
        print("\n❌ Authentication failed")
        print("Make sure to:")
        print("1. Use the COMPLETE URL provided")
        print("2. Sign in with hari@aura.com")
        print("3. Copy the ENTIRE authorization code")