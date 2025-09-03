#!/usr/bin/env python3
"""
GA4 OAuth using gcloud auth - Simpler approach
NO FALLBACKS - Real authentication
"""

import os
import json
import subprocess
from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
)
from google.auth import default
from google.oauth2 import service_account
import google.auth

# Configuration
GA_PROPERTY_ID = "308028264"

class GA4GCloudAuth:
    def __init__(self):
        self.client = None
        self.property = f"properties/{GA_PROPERTY_ID}"
        
    def authenticate_with_gcloud(self):
        """Use gcloud auth for authentication"""
        print("\n🔐 AUTHENTICATION OPTIONS")
        print("=" * 60)
        print("\nWe'll use gcloud auth to authenticate.")
        print("This will use your existing Google Cloud credentials.")
        print("=" * 60)
        
        # Check if gcloud is installed
        try:
            result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
            print("\n✅ gcloud is installed")
            print(result.stdout.split('\n')[0])
        except FileNotFoundError:
            print("\n❌ gcloud is not installed")
            print("Installing gcloud CLI...")
            # Install gcloud
            subprocess.run(['curl', 'https://sdk.cloud.google.com', '|', 'bash'], shell=True)
            return False
        
        # Check current auth
        result = subprocess.run(['gcloud', 'auth', 'list'], capture_output=True, text=True)
        print("\n📋 Current gcloud accounts:")
        print(result.stdout)
        
        if 'hari@aura.com' not in result.stdout:
            print("\n🔑 Need to authenticate with hari@aura.com")
            print("\nRun this command in your terminal:")
            print("\n  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/analytics.readonly")
            print("\nThis will open a browser where you can log in with hari@aura.com")
            print("\nAfter authenticating, run this script again.")
            return False
        
        # Use application default credentials
        try:
            credentials, project = default()
            self.client = BetaAnalyticsDataClient(credentials=credentials)
            print("\n✅ Authenticated with gcloud credentials")
            return True
        except Exception as e:
            print(f"\n❌ Authentication failed: {e}")
            print("\nTry running:")
            print("  gcloud auth application-default login")
            return False
    
    def authenticate_with_service_account(self):
        """use service account if available"""
        sa_path = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
        if sa_path.exists():
            print(f"\n🔑 Using service account: {sa_path}")
            credentials = service_account.Credentials.from_service_account_file(
                str(sa_path),
                scopes=['https://www.googleapis.com/auth/analytics.readonly']
            )
            self.client = BetaAnalyticsDataClient(credentials=credentials)
            return True
        return False
    
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
                dimensions=[Dimension(name="sessionDefaultChannelGroup")],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="conversions"),
                ],
                limit=10
            )
            
            response = self.client.run_report(request)
            
            print(f"\n✅ CONNECTION SUCCESSFUL!")
            print(f"Property ID: {GA_PROPERTY_ID}")
            print(f"\nLast 7 days overview:")
            
            total_sessions = 0
            total_conversions = 0
            
            for row in response.rows:
                channel = row.dimension_values[0].value or "(not set)"
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                total_sessions += sessions
                total_conversions += conversions
                
                if sessions > 0:
                    cvr = (conversions / sessions) * 100
                    print(f"  {channel}: {sessions:,} sessions, {conversions} conversions ({cvr:.2f}% CVR)")
            
            print(f"\n  TOTAL: {total_sessions:,} sessions, {total_conversions} conversions")
                
            return True
            
        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            
            if "403" in str(e) and "does not have sufficient permissions" in str(e):
                print("\n⚠️  ACCESS ISSUE DETECTED")
                print("=" * 60)
                print("\nThe service account needs to be added to GA4:")
                print("\n1. Go to Google Analytics")
                print("2. Admin → Property Access Management")
                print("3. Add this email: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com")
                print("4. Give it 'Viewer' role")
                print("\nOR use gcloud auth instead (with your hari@aura.com account)")
            
            return False
    
    def explore_data(self):
        """Explore GA4 data"""
        print("\n" + "="*60)
        print("GA4 DATA EXPLORATION")
        print("="*60)
        
        # Get top pages
        request = RunReportRequest(
            property=self.property,
            date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="pagePath"),
                Dimension(name="deviceCategory"),
            ],
            metrics=[
                Metric(name="screenPageViews"),
                Metric(name="conversions"),
                Metric(name="averageSessionDuration"),
            ],
            order_bys=[
                OrderBy(metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"), desc=True)
            ],
            limit=20
        )
        
        response = self.client.run_report(request)
        
        print("\n📱 TOP PAGES (Last 30 days):")
        
        for i, row in enumerate(response.rows[:10], 1):
            page = row.dimension_values[0].value or "(not set)"
            device = row.dimension_values[1].value or "(not set)"
            views = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            avg_duration = float(row.metric_values[2].value)
            
            # Highlight parental control pages
            if any(term in page.lower() for term in ['parent', 'control', 'family']):
                print(f"\n⭐ {i}. {page[:60]}...")
            else:
                print(f"\n{i}. {page[:60]}...")
            
            print(f"   Device: {device}")
            print(f"   Views: {views:,}")
            print(f"   Conversions: {conversions}")
            print(f"   Avg Duration: {avg_duration:.1f}s")

if __name__ == "__main__":
    connector = GA4GCloudAuth()
    
    print("\n🚀 GA4 AUTHENTICATION")
    print("=" * 60)
    print("Attempting to connect to GA4 property:", GA_PROPERTY_ID)
    print("=" * 60)
    
    # Try service account first
    authenticated = connector.authenticate_with_service_account()
    
    if not authenticated:
        # Try gcloud auth
        authenticated = connector.authenticate_with_gcloud()
    
    if authenticated:
        if connector.test_connection():
            connector.explore_data()
            print("\n✅ Successfully connected and explored GA4 data!")
        else:
            print("\n❌ Could not connect to GA4 property")
    else:
        print("\n❌ Authentication failed")
        print("\nNext steps:")
        print("1. Ask Jason to add the service account to GA4")
        print("2. OR run: gcloud auth application-default login")
        print("3. Then run this script again")