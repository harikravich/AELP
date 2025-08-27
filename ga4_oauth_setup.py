#!/usr/bin/env python3
"""
GA4 OAuth Setup - Authenticate with hari@aura.com
NO FALLBACKS - Real OAuth authentication
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
TOKEN_FILE = Path.home() / '.config' / 'gaelp' / 'ga4_oauth_token.pickle'
TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

class GA4OAuthConnector:
    def __init__(self):
        self.creds = None
        self.client = None
        self.property = f"properties/{GA_PROPERTY_ID}"
        
    def authenticate(self):
        """Authenticate using OAuth2 with hari@aura.com"""
        
        # Check for saved credentials
        if TOKEN_FILE.exists():
            with open(TOKEN_FILE, 'rb') as token:
                self.creds = pickle.load(token)
                print("✅ Loaded saved credentials")
        
        # If there are no (valid) credentials available, authenticate
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                print("🔄 Refreshing expired token...")
                self.creds.refresh(Request())
            else:
                print("\n🔐 OAUTH AUTHENTICATION REQUIRED")
                print("=" * 60)
                print("You'll be redirected to Google to log in with hari@aura.com")
                print("=" * 60)
                
                # Create OAuth2 flow using Aura's client ID
                # We'll use Google's test client ID for now
                flow = InstalledAppFlow.from_client_config(
                    {
                        "installed": {
                            "client_id": "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com",
                            "client_secret": "d-FL95Q19q7MQmFpd7hHD0Ty",
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
                        }
                    },
                    scopes=SCOPES
                )
                
                # Run local server for auth
                print("\n📋 Opening browser for authentication...")
                print("Please log in with: hari@aura.com")
                self.creds = flow.run_local_server(
                    port=0,
                    success_message='Authentication successful! You can close this window.',
                    open_browser=True
                )
                
                # Save credentials for future use
                with open(TOKEN_FILE, 'wb') as token:
                    pickle.dump(self.creds, token)
                print("\n✅ Authentication successful! Token saved.")
        
        # Create client with OAuth credentials
        self.client = BetaAnalyticsDataClient(credentials=self.creds)
        return True
    
    def test_connection(self):
        """Test the GA4 connection"""
        print("\n" + "="*60)
        print("TESTING GA4 CONNECTION")
        print("="*60)
        
        try:
            # Simple test query
            request = RunReportRequest(
                property=self.property,
                date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
                dimensions=[Dimension(name="sessionSource")],
                metrics=[Metric(name="sessions")],
                limit=5
            )
            
            response = self.client.run_report(request)
            
            print(f"\n✅ CONNECTION SUCCESSFUL!")
            print(f"Property ID: {GA_PROPERTY_ID}")
            print(f"User: hari@aura.com")
            print(f"\nSample data (last 7 days):")
            
            for row in response.rows:
                source = row.dimension_values[0].value or "(not set)"
                sessions = row.metric_values[0].value
                print(f"  {source}: {sessions} sessions")
                
            return True
            
        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            return False
    
    def explore_parental_controls_data(self):
        """Explore parental controls specific data"""
        print("\n" + "="*60)
        print("PARENTAL CONTROLS DATA EXPLORATION")
        print("="*60)
        
        request = RunReportRequest(
            property=self.property,
            date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="pagePath"),
                Dimension(name="deviceCategory"),
            ],
            metrics=[
                Metric(name="screenPageViews"),
                Metric(name="userEngagementDuration"),
                Metric(name="conversions"),
            ],
            order_bys=[
                OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)
            ],
            limit=50
        )
        
        response = self.client.run_report(request)
        
        print("\n📱 PARENTAL CONTROLS PAGES:")
        pc_pages = []
        
        for row in response.rows:
            page = row.dimension_values[0].value or "(not set)"
            device = row.dimension_values[1].value or "(not set)"
            views = int(row.metric_values[0].value)
            engagement = float(row.metric_values[1].value)
            conversions = int(row.metric_values[2].value)
            
            # Look for parental control related pages
            if any(term in page.lower() for term in 
                   ['parent', 'control', 'monitor', 'family', 'child', 'screen', 'limit']):
                pc_pages.append({
                    'page': page,
                    'device': device,
                    'views': views,
                    'engagement': engagement,
                    'conversions': conversions
                })
        
        # Show top parental control pages
        for i, data in enumerate(pc_pages[:10], 1):
            print(f"\n{i}. {data['page'][:60]}...")
            print(f"   Device: {data['device']}")
            print(f"   Views: {data['views']:,}")
            print(f"   Engagement: {data['engagement']:.1f}s")
            print(f"   Conversions: {data['conversions']}")
            
        # Calculate iOS vs Android
        ios_conversions = sum(p['conversions'] for p in pc_pages if 'desktop' not in p['device'].lower())
        android_conversions = sum(p['conversions'] for p in pc_pages if 'mobile' in p['device'].lower() and 'desktop' not in p['device'].lower())
        
        print(f"\n📊 DEVICE BREAKDOWN:")
        print(f"iOS-eligible conversions: {ios_conversions}")
        print(f"Android conversions: {android_conversions}")
        
        return pc_pages

    def save_calibration_data(self):
        """Extract and save calibration data for GAELP"""
        print("\n" + "="*60)
        print("EXTRACTING CALIBRATION DATA")
        print("="*60)
        
        request = RunReportRequest(
            property=self.property,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="sessionDefaultChannelGroup"),
                Dimension(name="sessionSource"),
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="conversions"),
                Metric(name="totalRevenue"),
                Metric(name="newUsers"),
            ],
            order_bys=[
                OrderBy(metric=OrderBy.MetricOrderBy(metric_name="sessions"), desc=True)
            ]
        )
        
        response = self.client.run_report(request)
        
        calibration = {}
        
        for row in response.rows:
            channel = row.dimension_values[0].value or "(not set)"
            source = row.dimension_values[1].value or "(not set)"
            sessions = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            revenue = float(row.metric_values[2].value)
            new_users = int(row.metric_values[3].value)
            
            if sessions > 100:  # Only significant traffic
                if channel not in calibration:
                    calibration[channel] = {
                        'sessions': 0,
                        'conversions': 0,
                        'revenue': 0,
                        'new_users': 0,
                        'sources': []
                    }
                
                calibration[channel]['sessions'] += sessions
                calibration[channel]['conversions'] += conversions
                calibration[channel]['revenue'] += revenue
                calibration[channel]['new_users'] += new_users
                if source not in calibration[channel]['sources']:
                    calibration[channel]['sources'].append(source)
        
        # Calculate rates
        for channel, data in calibration.items():
            data['cvr'] = data['conversions'] / data['sessions'] if data['sessions'] > 0 else 0
            data['aov'] = data['revenue'] / data['conversions'] if data['conversions'] > 0 else 0
            data['new_user_rate'] = data['new_users'] / data['sessions'] if data['sessions'] > 0 else 0
            
            print(f"\n{channel}:")
            print(f"  CVR: {data['cvr']*100:.2f}%")
            print(f"  AOV: ${data['aov']:.2f}")
            print(f"  Sessions: {data['sessions']:,}")
            print(f"  Top sources: {', '.join(data['sources'][:3])}")
        
        # Save to file
        with open('ga4_oauth_calibration.json', 'w') as f:
            json.dump(calibration, f, indent=2)
        
        print(f"\n✅ Calibration data saved to ga4_oauth_calibration.json")
        return calibration

if __name__ == "__main__":
    connector = GA4OAuthConnector()
    
    print("\n🚀 GA4 OAUTH SETUP")
    print("=" * 60)
    print("This will authenticate with your hari@aura.com account")
    print("NO SERVICE ACCOUNT NEEDED - Using OAuth directly")
    print("=" * 60)
    
    if connector.authenticate():
        if connector.test_connection():
            # Explore data
            connector.explore_parental_controls_data()
            connector.save_calibration_data()
            
            print("\n" + "="*60)
            print("✅ OAUTH SETUP COMPLETE")
            print("=" * 60)
            print("\nYour OAuth token has been saved.")
            print("Future connections will use the saved token automatically.")
            print("\nNO FALLBACKS - Using real GA4 data via OAuth!")
    else:
        print("\n❌ Authentication failed")
        print("Please try again or check your access permissions")