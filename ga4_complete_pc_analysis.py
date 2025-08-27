#!/usr/bin/env python3
"""
COMPLETE AND THOROUGH PC/Balance Analysis
No conclusions until all data is examined
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
print("COMPLETE PC/BALANCE PRODUCT ANALYSIS")
print("="*80)

# 1. ALL PRODUCTS WITH DETAILED BREAKDOWN
print("\n1. ALL PRODUCT VARIANTS (itemName):")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[Dimension(name="itemName")],
    metrics=[
        Metric(name="itemsPurchased"),
        Metric(name="itemRevenue"),
        Metric(name="itemsViewed"),
        Metric(name="itemsAddedToCart")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="itemsPurchased"))],
    limit=100
)

response = client.run_report(request)

all_products = []
pc_products = []
standalone_pc = []
bundle_pc = []

for row in response.rows:
    product = row.dimension_values[0].value or "(not set)"
    purchased = int(row.metric_values[0].value) if row.metric_values[0].value else 0
    revenue = float(row.metric_values[1].value) if row.metric_values[1].value else 0.0
    viewed = int(row.metric_values[2].value) if row.metric_values[2].value else 0
    added_cart = int(row.metric_values[3].value) if row.metric_values[3].value else 0
    
    if purchased > 0:
        avg_price = revenue / purchased if purchased > 0 else 0
        
        # Categorize products
        is_pc = any(kw in product.lower() for kw in ['pc', 'parent', 'balance', 'family', 'child', 'screen'])
        is_bundle = 'with' in product.lower() and 'pc' in product.lower()
        is_standalone = is_pc and not is_bundle
        
        product_data = {
            'name': product,
            'purchased': purchased,
            'revenue': revenue,
            'avg_price': avg_price,
            'viewed': viewed,
            'added_cart': added_cart
        }
        
        all_products.append(product_data)
        
        if is_pc:
            pc_products.append(product_data)
            if is_bundle:
                bundle_pc.append(product_data)
            elif is_standalone:
                standalone_pc.append(product_data)
        
        # Print all products
        marker = ""
        if is_standalone:
            marker = "🎯 STANDALONE PC:"
        elif is_bundle:
            marker = "📦 BUNDLE WITH PC:"
        
        if marker or purchased > 10:
            print(f"{marker:<20} {product[:40]:<40} | Sold: {purchased:,} | ${avg_price:.2f} avg | ${revenue:,.2f} total")

# 2. PLAN CODES ANALYSIS
print("\n2. PLAN CODES (customEvent:plan_code):")
print("-" * 80)

try:
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[Dimension(name="customEvent:plan_code")],
        metrics=[Metric(name="eventCount")],
        order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))],
        limit=50
    )
    
    response = client.run_report(request)
    
    for row in response.rows:
        plan = row.dimension_values[0].value or "(not set)"
        count = int(row.metric_values[0].value)
        
        if count > 100:
            is_pc = any(kw in plan.lower() for kw in ['pc', 'parent', 'balance', 'family'])
            marker = "🎯" if is_pc else "  "
            print(f"{marker} {plan:<40} {count:,} events")
            
except Exception as e:
    print(f"Plan code dimension not available: {e}")

# 3. OFFER CODES ANALYSIS
print("\n3. OFFER CODES (customEvent:offer_code):")
print("-" * 80)

try:
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[Dimension(name="customEvent:offer_code")],
        metrics=[Metric(name="conversions")],
        order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
        limit=50
    )
    
    response = client.run_report(request)
    
    for row in response.rows:
        offer = row.dimension_values[0].value or "(not set)"
        conversions = int(row.metric_values[0].value)
        
        if conversions > 10:
            is_pc = any(kw in offer.lower() for kw in ['pc', 'parent', 'balance', 'family'])
            marker = "🎯" if is_pc else "  "
            print(f"{marker} {offer:<40} {conversions:,} conversions")
            
except Exception as e:
    print(f"Offer code dimension not available: {e}")

# 4. ENROLLMENT PAGES WITH PRODUCTS
print("\n4. ENROLLMENT PAGE PRODUCT ANALYSIS:")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="pagePath"),
        Dimension(name="eventName")
    ],
    metrics=[Metric(name="eventCount")],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))],
    limit=200
)

response = client.run_report(request)

enrollment_purchases = {}
pc_page_purchases = {}

for row in response.rows:
    page = row.dimension_values[0].value or "(not set)"
    event = row.dimension_values[1].value
    count = int(row.metric_values[0].value)
    
    if event == "purchase":
        if "enrollment" in page:
            enrollment_purchases[page] = enrollment_purchases.get(page, 0) + count
        
        # Check for PC-specific enrollment pages
        if any(kw in page.lower() for kw in ['pc', 'parent', 'balance', 'family']):
            pc_page_purchases[page] = pc_page_purchases.get(page, 0) + count

print("Top Enrollment Pages with Purchases:")
for page, count in sorted(enrollment_purchases.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {page:<50} {count:,} purchases")

if pc_page_purchases:
    print("\nPC-Specific Enrollment Pages:")
    for page, count in sorted(pc_page_purchases.items(), key=lambda x: x[1], reverse=True):
        print(f"  🎯 {page:<50} {count:,} purchases")

# 5. GATEWAY ANALYSIS (how products are categorized)
print("\n5. GATEWAY CATEGORIZATION (customEvent:gateway):")
print("-" * 80)

try:
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
        dimensions=[Dimension(name="customEvent:gateway")],
        metrics=[
            Metric(name="conversions"),
            Metric(name="totalUsers")
        ],
        order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
        limit=30
    )
    
    response = client.run_report(request)
    
    for row in response.rows:
        gateway = row.dimension_values[0].value or "(not set)"
        conversions = int(row.metric_values[0].value)
        users = int(row.metric_values[1].value)
        
        if conversions > 10:
            cvr = (conversions / users * 100) if users > 0 else 0
            is_pc = any(kw in gateway.lower() for kw in ['pc', 'parent', 'balance', 'family'])
            marker = "🎯" if is_pc else "  "
            print(f"{marker} {gateway:<30} {conversions:,} conv | {users:,} users | {cvr:.2f}% CVR")
            
except Exception as e:
    print(f"Gateway dimension error: {e}")

# 6. FAMILY OFFER CODES (specific to PC?)
print("\n6. FAMILY/PC SPECIFIC OFFERS:")
print("-" * 80)

family_dimensions = [
    "customEvent:family_offer_code",
    "customEvent:couple_offer_code",
    "customEvent:individual_offer_code"
]

for dim in family_dimensions:
    try:
        request = RunReportRequest(
            property=f"properties/{GA_PROPERTY_ID}",
            date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
            dimensions=[Dimension(name=dim)],
            metrics=[Metric(name="conversions")],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
            limit=10
        )
        
        response = client.run_report(request)
        
        if response.row_count > 0:
            print(f"\n{dim}:")
            for row in response.rows:
                value = row.dimension_values[0].value or "(not set)"
                conversions = int(row.metric_values[0].value)
                if conversions > 0 and value != "(not set)":
                    print(f"  {value:<30} {conversions:,} conversions")
                    
    except:
        pass

# 7. SUMMARY STATISTICS
print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("-" * 80)

total_purchases = sum(p['purchased'] for p in all_products)
pc_total = sum(p['purchased'] for p in pc_products)
standalone_total = sum(p['purchased'] for p in standalone_pc)
bundle_total = sum(p['purchased'] for p in bundle_pc)

print(f"\nTotal Products Purchased: {total_purchases:,}")
print(f"PC-Related Products: {pc_total:,} ({pc_total/total_purchases*100:.1f}% of total)")
print(f"  - Standalone PC: {standalone_total:,}")
print(f"  - Bundle with PC: {bundle_total:,}")
print(f"  - Other PC variants: {pc_total - standalone_total - bundle_total:,}")

if pc_products:
    print("\nPC Product Details:")
    for p in pc_products:
        print(f"  {p['name'][:50]:<50} | {p['purchased']:,} sold | ${p['avg_price']:.2f}")

print("\n" + "="*80)
print("DATA COLLECTION COMPLETE - NO PREMATURE CONCLUSIONS")
print("Waiting for full analysis before determining product structure...")