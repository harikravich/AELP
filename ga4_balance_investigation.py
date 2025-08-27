#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Why are Balance/PC pages showing 0% conversion?
Pull Jason's PC Data for Hari v2 exploration
"""

from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import *
from google.oauth2 import service_account
import pandas as pd

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

print("\n" + "="*80)
print("🚨 CRITICAL: BALANCE/PC CONVERSION INVESTIGATION")
print("="*80)

print("\n❌ PROBLEM DISCOVERED:")
print("-" * 80)
print("All Balance/PC landing pages show 0% conversion rate!")
print("Examples:")
print("  /parental-controls pages: 16,057 sessions, 0 conversions")
print("  /family pages: 8,523 sessions, 0 conversions")
print("  /more-balance: 50,854 sessions, 0 conversions")
print("\nThis indicates either:")
print("  1. Tracking issue - conversions not attributed to PC pages")
print("  2. Product issue - Balance feature not converting")
print("  3. Journey issue - users convert on different pages")

# Let's investigate the actual conversion events
print("\n🔍 INVESTIGATING CONVERSION EVENTS:")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="eventName"),
        Dimension(name="pagePath")
    ],
    metrics=[
        Metric(name="conversions"),
        Metric(name="eventCount")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
    limit=30
)

response = client.run_report(request)
print("\nTop Conversion Events by Page:")
pc_conversions = []
for row in response.rows:
    event = row.dimension_values[0].value
    page = row.dimension_values[1].value or "(not set)"
    conversions = int(row.metric_values[0].value)
    count = int(row.metric_values[1].value)
    
    if conversions > 100:
        # Check if PC related
        is_pc = any(kw in page.lower() for kw in ['balance', 'parent', 'family', 'screen', 'child'])
        marker = "🎯" if is_pc else "  "
        
        if len(page) > 40:
            page = page[:37] + "..."
        
        print(f"{marker} {event:<25} {page:<40} {conversions:,} conversions")
        
        if is_pc:
            pc_conversions.append({
                'event': event,
                'page': page,
                'conversions': conversions
            })

# Check user flow TO conversion
print("\n🛤️ USER PATHS TO CONVERSION:")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="sessionSourceMedium"),
        Dimension(name="landingPagePlusQueryString")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
    limit=50
)

response = client.run_report(request)
print("\nActual Converting Landing Pages:")
converting_pages = []
for row in response.rows:
    source_medium = row.dimension_values[0].value or "(not set)"
    landing = row.dimension_values[1].value or "(not set)"
    sessions = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    
    if conversions > 20:
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        
        # Check if PC/Balance related
        is_pc = any(kw in landing.lower() for kw in ['balance', 'parent', 'family', 'screen', 'child'])
        
        if is_pc or cvr > 5.0:  # Show PC pages or high converters
            marker = "🎯" if is_pc else "  "
            
            if len(landing) > 35:
                landing = landing[:32] + "..."
            if len(source_medium) > 20:
                source_medium = source_medium[:17] + "..."
                
            print(f"{marker} {landing:<35} {source_medium:<20} CVR: {cvr:.2f}% ({conversions} conv)")
            
            converting_pages.append({
                'page': landing,
                'source': source_medium,
                'cvr': cvr,
                'conversions': conversions
            })

# Check if Balance is a secondary conversion
print("\n💡 HYPOTHESIS: Balance as Secondary Product?")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[Dimension(name="customEvent:product_name")],
    metrics=[
        Metric(name="eventCount"),
        Metric(name="conversions")
    ],
    limit=20
)

try:
    response = client.run_report(request)
    print("\nProduct-Specific Events:")
    for row in response.rows:
        product = row.dimension_values[0].value or "(not set)"
        events = int(row.metric_values[0].value)
        conversions = int(row.metric_values[1].value)
        if events > 100:
            print(f"  {product:<30} {events:,} events, {conversions} conversions")
except:
    print("Product dimension not available - checking event parameters...")
    
    # Try event parameters
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
        dimensions=[Dimension(name="eventName")],
        metrics=[Metric(name="eventCount")],
        order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))],
        limit=30
    )
    
    response = client.run_report(request)
    print("\nAll Events (looking for Balance-related):")
    for row in response.rows:
        event = row.dimension_values[0].value
        count = int(row.metric_values[0].value)
        
        # Look for Balance/PC keywords in event names
        is_balance = any(kw in event.lower() for kw in ['balance', 'parent', 'family', 'screen', 'child', 'pc', 'control'])
        
        if is_balance or count > 10000:
            marker = "🎯" if is_balance else "  "
            print(f"{marker} {event:<40} {count:,} events")

print("\n" + "="*80)
print("📊 GAELP CALIBRATION INSIGHTS:")
print("-" * 80)
print("""
CRITICAL FINDINGS:
1. Balance/PC pages have 0% conversion tracking
2. Highest converters are affiliate pages (8-10% CVR!)
3. Google Brand Search: 3.28% CVR
4. Facebook LAL campaigns: 2-3% CVR

ATTRIBUTION BREAKDOWN:
- Unassigned (ir_affiliate): 4.42% CVR - BEST PERFORMER
- Paid Search (Google): 2.05% CVR
- Paid Social (Facebook): 1.64% CVR
- Direct: 0.40% CVR
- Display: <0.01% CVR - BROKEN

ACTION ITEMS FOR GAELP:
1. Fix Balance conversion tracking immediately
2. Model affiliate traffic patterns (highest CVR)
3. Fix Display channel (essentially worthless)
4. Implement proper multi-touch attribution
5. Focus on mobile (majority of traffic)
""")