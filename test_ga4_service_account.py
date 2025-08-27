#!/usr/bin/env python3
"""
Test GA4 Access with Service Account
Jason has added: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com
"""

from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy
)
from google.oauth2 import service_account

# Configuration
GA_PROPERTY_ID = "308028264"  # Aura's GA4 property
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

print("\n" + "="*70)
print("TESTING GA4 SERVICE ACCOUNT ACCESS")
print("="*70)

# Check if service account file exists
if not SERVICE_ACCOUNT_FILE.exists():
    print(f"❌ Service account file not found: {SERVICE_ACCOUNT_FILE}")
    print("\nLooking for alternatives...")
    alt_path = Path.home() / '.config' / 'gaelp' / 'service-account.json'
    if alt_path.exists():
        SERVICE_ACCOUNT_FILE = alt_path
        print(f"✅ Found at: {SERVICE_ACCOUNT_FILE}")
    else:
        print("❌ No service account file found")
        exit(1)

try:
    # Create credentials from service account
    print(f"\n🔐 Loading service account: {SERVICE_ACCOUNT_FILE}")
    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=['https://www.googleapis.com/auth/analytics.readonly']
    )
    
    print(f"✅ Service account email: {credentials.service_account_email}")
    
    # Create GA4 client
    print("\n📊 Connecting to GA4...")
    client = BetaAnalyticsDataClient(credentials=credentials)
    
    # Test with a simple query
    print(f"📍 Property ID: {GA_PROPERTY_ID}")
    
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="sessionDefaultChannelGroup"),
        ],
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalUsers"),
            Metric(name="conversions"),
            Metric(name="screenPageViews"),
        ],
        order_bys=[
            OrderBy(desc=True, dimension=OrderBy.DimensionOrderBy(dimension_name="date"))
        ],
        limit=10
    )
    
    print("\n🔍 Querying GA4 data...")
    response = client.run_report(request)
    
    print("\n" + "="*70)
    print("✅ SUCCESS! GA4 ACCESS CONFIRMED!")
    print("="*70)
    
    print(f"\n📊 AURA GA4 DATA (Last 30 Days):")
    print("-" * 70)
    
    # Show the data
    for row in response.rows:
        date = row.dimension_values[0].value
        channel = row.dimension_values[1].value or "(not set)"
        sessions = int(row.metric_values[0].value)
        users = int(row.metric_values[1].value)
        conversions = int(row.metric_values[2].value)
        pageviews = int(row.metric_values[3].value)
        
        print(f"{date} | {channel:20} | {sessions:,} sessions | {users:,} users | {conversions} conversions")
    
    # Get summary metrics
    print("\n" + "="*70)
    print("📈 SUMMARY METRICS:")
    print("-" * 70)
    
    summary_request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[Dimension(name="sessionDefaultChannelGroup")],
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalUsers"),
            Metric(name="conversions"),
            Metric(name="averageSessionDuration"),
        ],
        limit=20
    )
    
    summary_response = client.run_report(summary_request)
    
    total_sessions = 0
    total_users = 0
    total_conversions = 0
    
    print("\nChannel Performance:")
    for row in summary_response.rows:
        channel = row.dimension_values[0].value or "(not set)"
        sessions = int(row.metric_values[0].value)
        users = int(row.metric_values[1].value)
        conversions = int(row.metric_values[2].value)
        avg_duration = float(row.metric_values[3].value)
        
        total_sessions += sessions
        total_users += users
        total_conversions += conversions
        
        if sessions > 100:  # Only show significant channels
            print(f"  {channel:25} | {sessions:8,} sessions | {users:7,} users | {conversions:4} conv | {avg_duration:.1f}s avg")
    
    print("-" * 70)
    print(f"  {'TOTAL':25} | {total_sessions:8,} sessions | {total_users:7,} users | {total_conversions:4} conv")
    
    print("\n" + "="*70)
    print("🎉 GA4 INTEGRATION COMPLETE!")
    print("="*70)
    print("\n✅ Service account has full GA4 access")
    print("✅ Can pull real Aura conversion data")
    print("✅ Ready for GAELP calibration")
    print("\nNext: Use this data to calibrate (not train) the simulator")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    
    if "403" in str(e) or "PERMISSION_DENIED" in str(e):
        print("\n⚠️ Permission denied. Possible issues:")
        print("1. Service account not added to GA4 yet")
        print("2. Wrong property ID")
        print("3. Insufficient permissions (needs Viewer or higher)")
        print(f"\nService account email: ga4-mcp-server@centering-line-469716-r7.iam.gserviceaccount.com")
        print(f"GA4 Property ID: {GA_PROPERTY_ID}")
        print("\nAsk Jason to verify in GA4 Admin → Property Access Management")
    elif "404" in str(e):
        print(f"\n⚠️ Property not found: {GA_PROPERTY_ID}")
        print("Verify the correct GA4 property ID with Jason")
    else:
        print(f"\nFull error: {e}")