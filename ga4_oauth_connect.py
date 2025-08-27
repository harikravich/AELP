#!/usr/bin/env python3
"""
GA4 OAuth Connection - Connect using hari@aura.com credentials
This uses OAuth flow instead of service account
"""

import os
import json
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.exceptions import RefreshError

# GA4 Property ID - we'll need to get this
GA_PROPERTY_ID = "308028264"  # From the URL you were given: p308028264

class GA4Explorer:
    def __init__(self):
        self.creds = None
        self.client = None
        
    def authenticate(self):
        """Authenticate using OAuth2 flow"""
        # Define the scopes
        SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
        
        # Token file to store credentials
        token_file = 'ga4_token.json'
        
        # Check if we have saved credentials
        if os.path.exists(token_file):
            self.creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                try:
                    self.creds.refresh(Request())
                except RefreshError:
                    os.remove(token_file)
                    print("Token expired. Please run authentication again.")
                    return False
            else:
                # Create OAuth client configuration
                client_config = {
                    "installed": {
                        "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
                        "client_secret": "YOUR_CLIENT_SECRET",
                        "redirect_uris": ["http://localhost:8080"],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token"
                    }
                }
                
                flow = Flow.from_client_config(
                    client_config,
                    scopes=SCOPES,
                    redirect_uri='http://localhost:8080'
                )
                
                # Get authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                
                print(f"Please visit this URL to authorize the application:")
                print(auth_url)
                
                # Get authorization code from user
                code = input("Enter the authorization code: ")
                
                # Exchange code for tokens
                flow.fetch_token(code=code)
                self.creds = flow.credentials
                
                # Save credentials for next run
                with open(token_file, 'w') as token:
                    token.write(self.creds.to_json())
        
        # Create client with credentials
        self.client = BetaAnalyticsDataClient(credentials=self.creds)
        return True
    
    def explore_data(self):
        """Explore available GA4 data"""
        if not self.client:
            print("Not authenticated. Run authenticate() first.")
            return
        
        # Create a simple report request
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="sessionSource"),
                Dimension(name="sessionMedium"),
                Dimension(name="deviceCategory"),
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="newUsers"),
                Metric(name="conversions"),
                Metric(name="userEngagementDuration"),
            ],
        )
        
        try:
            response = self.client.run_report(request)
            
            print("\n=== GA4 Data Summary (Last 30 Days) ===\n")
            print(f"Property ID: {GA_PROPERTY_ID}")
            print(f"Total rows: {len(response.rows)}")
            
            print("\n=== Top Traffic Sources ===")
            for row in response.rows[:10]:
                source = row.dimension_values[0].value
                medium = row.dimension_values[1].value
                device = row.dimension_values[2].value
                sessions = row.metric_values[0].value
                users = row.metric_values[1].value
                conversions = row.metric_values[4].value if len(row.metric_values) > 4 else "0"
                
                print(f"Source: {source}/{medium} ({device})")
                print(f"  Sessions: {sessions}, Users: {users}, Conversions: {conversions}")
                print()
                
        except Exception as e:
            print(f"Error accessing GA4 data: {e}")
            print("\nThis might be because:")
            print("1. The property ID is incorrect")
            print("2. You don't have access to this property")
            print("3. The API is not enabled")

if __name__ == "__main__":
    explorer = GA4Explorer()
    
    print("GA4 OAuth Connection Explorer")
    print("=" * 40)
    print("\nNote: This will open a browser for authentication")
    print("You'll need to log in with hari@aura.com\n")
    
    if explorer.authenticate():
        print("Authentication successful!")
        explorer.explore_data()
    else:
        print("Authentication failed.")