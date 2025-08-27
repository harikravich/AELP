#!/usr/bin/env python3
"""
Understand GA4 Hierarchy and Setup for Aura
Pull the actual PC Data for Hari v2 report structure
"""

from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import *
from google.oauth2 import service_account
import json

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

print("\n" + "="*80)
print("🔍 GA4 HIERARCHY & SETUP INVESTIGATION")
print("="*80)

# 1. Check available custom dimensions
print("\n📊 CHECKING CUSTOM DIMENSIONS & EVENTS:")
print("-" * 80)

# Get metadata about the property
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import GetMetadataRequest

metadata_request = GetMetadataRequest(name=f"properties/{GA_PROPERTY_ID}/metadata")

try:
    metadata = client.get_metadata(metadata_request)
    
    print("\nCustom Dimensions Available:")
    custom_dims = []
    for dim in metadata.dimensions:
        if 'custom' in dim.api_name.lower() or 'product' in dim.api_name.lower() or 'plan' in dim.api_name.lower():
            print(f"  {dim.api_name:<40} {dim.ui_name}")
            custom_dims.append(dim.api_name)
    
    print("\nProduct/Plan Related Metrics:")
    for metric in metadata.metrics:
        if any(kw in metric.api_name.lower() for kw in ['product', 'plan', 'purchase', 'revenue', 'conversion']):
            print(f"  {metric.api_name:<40} {metric.ui_name}")
            
except Exception as e:
    print(f"Metadata error: {e}")

# 2. Check how products are tracked
print("\n🛍️ PRODUCT/PLAN TRACKING:")
print("-" * 80)

# Try different product-related dimensions
product_dimensions = [
    "itemName",
    "itemId", 
    "itemCategory",
    "itemCategory2",
    "itemCategory3",
    "itemCategory4",
    "itemCategory5",
    "itemBrand",
    "itemVariant"
]

for dim in product_dimensions:
    try:
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
            dimensions=[Dimension(name=dim)],
            metrics=[Metric(name="itemsViewed"), Metric(name="itemsPurchased")],
            limit=20
        )
        
        response = client.run_report(request)
        
        if response.row_count > 0:
            print(f"\n{dim} (Product Dimension):")
            for row in response.rows:
                item = row.dimension_values[0].value or "(not set)"
                viewed = int(row.metric_values[0].value) if row.metric_values[0].value else 0
                purchased = int(row.metric_values[1].value) if row.metric_values[1].value else 0
                
                if viewed > 100 or purchased > 10:
                    # Check if Balance/PC related
                    is_balance = any(kw in item.lower() for kw in ['balance', 'parent', 'family', 'screen', 'control', 'child'])
                    marker = "🎯" if is_balance else "  "
                    print(f"{marker} {item:<40} Viewed: {viewed:,} | Purchased: {purchased:,}")
                    
    except Exception as e:
        if "not found" not in str(e).lower():
            print(f"Error with {dim}: {e}")

# 3. Check ecommerce events
print("\n💳 ECOMMERCE EVENT TRACKING:")
print("-" * 80)

ecommerce_events = [
    "purchase",
    "add_to_cart",
    "begin_checkout",
    "view_item",
    "view_item_list",
    "select_item",
    "add_payment_info",
    "add_shipping_info"
]

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[Dimension(name="eventName")],
    metrics=[Metric(name="eventCount"), Metric(name="conversions")],
    limit=100
)

response = client.run_report(request)

print("\nEcommerce Events Found:")
for row in response.rows:
    event = row.dimension_values[0].value
    count = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    
    if event in ecommerce_events or 'purchase' in event.lower() or 'checkout' in event.lower():
        print(f"  {event:<30} {count:,} events | {conversions:,} conversions")

# 4. Check user properties
print("\n👤 USER PROPERTIES & SEGMENTS:")
print("-" * 80)

# Try to find user properties related to products
user_properties = [
    "customUser:user_id",
    "customUser:customer_type",
    "customUser:subscription_status",
    "customUser:product_suite",
    "customUser:has_balance",
    "customUser:has_parental_controls"
]

for prop in user_properties:
    try:
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
            dimensions=[Dimension(name=prop)],
            metrics=[Metric(name="activeUsers")],
            limit=10
        )
        
        response = client.run_report(request)
        
        if response.row_count > 0:
            print(f"\n{prop}:")
            for row in response.rows:
                value = row.dimension_values[0].value or "(not set)"
                users = int(row.metric_values[0].value)
                print(f"  {value:<30} {users:,} users")
                
    except:
        pass  # User property doesn't exist

# 5. Check how conversions are defined
print("\n🎯 CONVERSION DEFINITION:")
print("-" * 80)

# Get all events marked as conversions
request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[Dimension(name="eventName")],
    metrics=[Metric(name="conversions")],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
    limit=20
)

response = client.run_report(request)

print("\nEvents Marked as Conversions:")
total_conversions = 0
for row in response.rows:
    event = row.dimension_values[0].value
    conversions = int(row.metric_values[0].value)
    
    if conversions > 0:
        total_conversions += conversions
        print(f"  {event:<30} {conversions:,} conversions")

print(f"\nTotal Conversions (30 days): {total_conversions:,}")

# 6. Check if Balance is an add-on or separate product
print("\n📦 PRODUCT HIERARCHY:")
print("-" * 80)

# Look at purchase events with item details
request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="eventName"),
        Dimension(name="pagePath")
    ],
    metrics=[Metric(name="eventCount")],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))],
    limit=100
)

response = client.run_report(request)

print("\nBalance/PC Related Events:")
for row in response.rows:
    event = row.dimension_values[0].value
    page = row.dimension_values[1].value or "(not set)"
    count = int(row.metric_values[0].value)
    
    # Look for Balance/PC indicators
    if any(kw in event.lower() or kw in page.lower() for kw in ['balance', 'parent', 'family', 'screen', 'control', 'child', 'youth']):
        if len(page) > 40:
            page = page[:37] + "..."
        print(f"  {event:<25} {page:<40} {count:,} events")

print("\n" + "="*80)
print("💡 INSIGHTS:")
print("-" * 80)
print("""
HYPOTHESIS: Balance/PC might be:
1. An add-on to main Aura subscription (not tracked separately)
2. Part of a bundle (tracked under main product)
3. Using custom dimensions we haven't discovered yet
4. Tracked through user properties rather than events

The "PC Data for Hari v2" report likely uses:
- Custom segments for PC users
- User properties to identify Balance subscribers
- Cohort analysis to track PC feature adoption
- Multi-product attribution models

We need to understand:
- How Aura defines a "PC conversion"
- Whether Balance is a feature, add-on, or separate SKU
- What custom dimensions Jason's report uses
""")