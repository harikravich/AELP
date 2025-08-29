#!/usr/bin/env python3
"""
Analyze Balance (Parental Controls) specific campaigns and conversion rates
"""

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric, OrderBy
from google.oauth2 import service_account
from pathlib import Path
import pandas as pd
import json

print("="*80)
print("BALANCE (PARENTAL CONTROLS) CAMPAIGN ANALYSIS")
print("="*80)

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
OUTPUT_DIR = Path("/home/hariravichandran/AELP/data/balance_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

if not SERVICE_ACCOUNT_FILE.exists():
    SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)
client = BetaAnalyticsDataClient(credentials=credentials)

# 1. GET ALL BALANCE-RELATED CAMPAIGNS
print("\n📊 1. Fetching Balance-related campaigns...")
print("-" * 80)

campaign_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="90daysAgo", end_date="yesterday")],  # 3 months for better data
    dimensions=[
        Dimension(name="sessionCampaignName"),
        Dimension(name="sessionSource"),
        Dimension(name="sessionMedium"),
        Dimension(name="deviceCategory")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="totalUsers"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue"),
        Metric(name="addToCarts"),
        Metric(name="checkouts"),
        Metric(name="bounceRate"),
        Metric(name="engagementRate")
    ],
    order_bys=[
        OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))
    ],
    limit=500
)

response = client.run_report(campaign_request)

# Collect all campaigns
all_campaigns = []
balance_campaigns = []

for row in response.rows:
    campaign_name = row.dimension_values[0].value or '(not set)'
    source = row.dimension_values[1].value or '(direct)'
    medium = row.dimension_values[2].value or '(none)'
    device = row.dimension_values[3].value or 'desktop'
    
    campaign_data = {
        'campaign': campaign_name,
        'source': source,
        'medium': medium,
        'device': device,
        'sessions': int(row.metric_values[0].value),
        'users': int(row.metric_values[1].value),
        'conversions': int(row.metric_values[2].value),
        'revenue': float(row.metric_values[3].value) if row.metric_values[3].value else 0,
        'add_to_carts': int(row.metric_values[4].value) if row.metric_values[4].value else 0,
        'checkouts': int(row.metric_values[5].value) if row.metric_values[5].value else 0,
        'bounce_rate': float(row.metric_values[6].value) if row.metric_values[6].value else 0,
        'engagement_rate': float(row.metric_values[7].value) if row.metric_values[7].value else 0
    }
    
    campaign_data['cvr'] = campaign_data['conversions'] / campaign_data['sessions'] if campaign_data['sessions'] > 0 else 0
    campaign_data['aov'] = campaign_data['revenue'] / campaign_data['conversions'] if campaign_data['conversions'] > 0 else 0
    campaign_data['cart_to_checkout'] = campaign_data['checkouts'] / campaign_data['add_to_carts'] if campaign_data['add_to_carts'] > 0 else 0
    campaign_data['checkout_to_purchase'] = campaign_data['conversions'] / campaign_data['checkouts'] if campaign_data['checkouts'] > 0 else 0
    
    all_campaigns.append(campaign_data)
    
    # Check if this is Balance-related
    balance_keywords = ['balance', 'parent', 'family', 'child', 'kid', 'screen', 'monitor', 
                       'control', 'safe', 'teen', 'daughter', 'son', 'limit', 'block']
    
    if any(keyword in campaign_name.lower() for keyword in balance_keywords):
        balance_campaigns.append(campaign_data)

# Create DataFrames
all_df = pd.DataFrame(all_campaigns)
balance_df = pd.DataFrame(balance_campaigns) if balance_campaigns else pd.DataFrame()

# Save data
all_df.to_csv(OUTPUT_DIR / "all_campaigns.csv", index=False)
if not balance_df.empty:
    balance_df.to_csv(OUTPUT_DIR / "balance_campaigns.csv", index=False)

print(f"✅ Found {len(all_campaigns)} total campaigns")
print(f"✅ Found {len(balance_campaigns)} Balance-related campaigns")

if not balance_df.empty:
    # Show Balance campaign performance
    print("\n📈 TOP BALANCE CAMPAIGNS BY SESSIONS:")
    print("-" * 80)
    print(f"{'Campaign':<40} {'Sessions':>10} {'CVR':>8} {'AOV':>8} {'Revenue':>10}")
    print("-" * 80)
    
    for _, row in balance_df.nlargest(10, 'sessions').iterrows():
        print(f"{row['campaign'][:39]:<40} {row['sessions']:>10,} {row['cvr']*100:>7.2f}% ${row['aov']:>7.2f} ${row['revenue']:>10.2f}")
    
    # Aggregate Balance performance
    print("\n📊 BALANCE OVERALL PERFORMANCE:")
    print("-" * 80)
    
    total_sessions = balance_df['sessions'].sum()
    total_conversions = balance_df['conversions'].sum()
    total_revenue = balance_df['revenue'].sum()
    total_users = balance_df['users'].sum()
    total_add_to_carts = balance_df['add_to_carts'].sum()
    total_checkouts = balance_df['checkouts'].sum()
    
    overall_cvr = total_conversions / total_sessions if total_sessions > 0 else 0
    overall_aov = total_revenue / total_conversions if total_conversions > 0 else 0
    cart_rate = total_add_to_carts / total_sessions if total_sessions > 0 else 0
    
    print(f"Total Sessions:        {total_sessions:,}")
    print(f"Total Users:           {total_users:,}")
    print(f"Total Conversions:     {total_conversions:,}")
    print(f"Total Revenue:         ${total_revenue:,.2f}")
    print(f"Overall CVR:           {overall_cvr*100:.2f}%")
    print(f"Overall AOV:           ${overall_aov:.2f}")
    print(f"Add to Cart Rate:      {cart_rate*100:.2f}%")
    
    # Funnel analysis
    print("\n🔄 BALANCE CONVERSION FUNNEL:")
    print("-" * 80)
    print(f"Sessions:              {total_sessions:,} (100.0%)")
    print(f"Add to Cart:           {total_add_to_carts:,} ({total_add_to_carts/total_sessions*100:.1f}%)")
    print(f"Checkout Started:      {total_checkouts:,} ({total_checkouts/total_sessions*100:.1f}%)")
    print(f"Purchase Completed:    {total_conversions:,} ({overall_cvr*100:.1f}%)")
    
    if total_checkouts > 0:
        checkout_abandonment = 1 - (total_conversions / total_checkouts)
        print(f"\n⚠️ Checkout Abandonment Rate: {checkout_abandonment*100:.1f}%")
    
    # Device breakdown for Balance
    print("\n📱 BALANCE PERFORMANCE BY DEVICE:")
    print("-" * 80)
    
    device_summary = balance_df.groupby('device').agg({
        'sessions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    })
    device_summary['cvr'] = device_summary['conversions'] / device_summary['sessions']
    device_summary['aov'] = device_summary['revenue'] / device_summary['conversions'].replace(0, 1)
    
    for device, data in device_summary.iterrows():
        print(f"{device:15} | Sessions: {data['sessions']:8,} | CVR: {data['cvr']*100:6.2f}% | AOV: ${data['aov']:7.2f}")
    
    # Source/Medium breakdown
    print("\n🌐 BALANCE PERFORMANCE BY SOURCE:")
    print("-" * 80)
    
    source_summary = balance_df.groupby('source').agg({
        'sessions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).nlargest(10, 'sessions')
    
    source_summary['cvr'] = source_summary['conversions'] / source_summary['sessions']
    
    for source, data in source_summary.iterrows():
        if data['sessions'] > 100:  # Only show significant sources
            print(f"{source[:30]:<30} | Sessions: {data['sessions']:8,} | CVR: {data['cvr']*100:6.2f}%")

# 2. ANALYZE LANDING PAGES FOR BALANCE
print("\n📊 2. Analyzing Balance landing pages...")
print("-" * 80)

landing_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="landingPagePlusQueryString"),
        Dimension(name="sessionCampaignName")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="bounceRate"),
        Metric(name="averageSessionDuration")
    ],
    limit=200
)

landing_response = client.run_report(landing_request)

balance_landing_pages = []
for row in landing_response.rows:
    landing_page = row.dimension_values[0].value or '/'
    campaign = row.dimension_values[1].value or '(not set)'
    
    # Check if Balance-related
    if any(keyword in landing_page.lower() or keyword in campaign.lower() 
          for keyword in ['balance', 'parent', 'family', 'child', 'screen']):
        balance_landing_pages.append({
            'landing_page': landing_page[:100],  # Truncate long URLs
            'campaign': campaign[:50],
            'sessions': int(row.metric_values[0].value),
            'conversions': int(row.metric_values[1].value),
            'bounce_rate': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
            'avg_duration': float(row.metric_values[3].value) if row.metric_values[3].value else 0
        })

if balance_landing_pages:
    landing_df = pd.DataFrame(balance_landing_pages)
    landing_df['cvr'] = landing_df['conversions'] / landing_df['sessions'].replace(0, 1)
    
    print("\n🎯 TOP BALANCE LANDING PAGES:")
    print("-" * 80)
    
    top_landings = landing_df.nlargest(5, 'sessions')
    for _, row in top_landings.iterrows():
        print(f"Page: {row['landing_page'][:50]}")
        print(f"  Sessions: {row['sessions']:,} | CVR: {row['cvr']*100:.2f}% | Bounce: {row['bounce_rate']:.1f}%")
        print()

# 3. COMPARISON WITH OTHER PRODUCTS
print("\n📊 3. Balance vs Other Products Comparison...")
print("-" * 80)

# Define product categories
product_keywords = {
    'balance/parental': ['balance', 'parent', 'family', 'child', 'screen', 'teen', 'kid'],
    'vpn': ['vpn', 'privacy', 'secure', 'encrypt'],
    'antivirus': ['virus', 'malware', 'protect', 'scan'],
    'identity': ['identity', 'theft', 'credit', 'alert'],
    'password': ['password', 'vault', 'credential']
}

product_performance = {}
for product, keywords in product_keywords.items():
    product_campaigns = all_df[all_df['campaign'].str.lower().str.contains('|'.join(keywords), na=False)]
    if not product_campaigns.empty:
        product_performance[product] = {
            'sessions': product_campaigns['sessions'].sum(),
            'conversions': product_campaigns['conversions'].sum(),
            'revenue': product_campaigns['revenue'].sum()
        }

print("\n🏆 PRODUCT COMPARISON:")
print("-" * 80)
print(f"{'Product':<20} {'Sessions':>12} {'Conversions':>12} {'CVR':>8} {'Revenue':>12}")
print("-" * 80)

for product, data in product_performance.items():
    cvr = data['conversions'] / data['sessions'] if data['sessions'] > 0 else 0
    print(f"{product:<20} {data['sessions']:>12,} {data['conversions']:>12,} {cvr*100:>7.2f}% ${data['revenue']:>11,.2f}")

print("\n" + "="*80)
print("KEY FINDINGS FOR BALANCE (PARENTAL CONTROLS)")
print("="*80)

if not balance_df.empty:
    print(f"""
✅ Balance Campaign Performance:
   - Total Sessions: {total_sessions:,}
   - Overall CVR: {overall_cvr*100:.2f}%
   - Average Order Value: ${overall_aov:.2f}
   - Total Revenue: ${total_revenue:,.2f}

📊 Conversion Funnel Issues:
   - Add to Cart Rate: {cart_rate*100:.2f}%
   - Checkout Abandonment: {(1 - total_conversions/total_checkouts)*100 if total_checkouts > 0 else 0:.1f}%
   
🎯 Recommendations:
   - Focus on checkout abandonment reduction
   - Test different pricing/messaging for parents
   - Optimize mobile experience (highest traffic source)
""")
else:
    print("\n⚠️ No Balance-specific campaigns found in the data")

# Save summary
summary = {
    'total_campaigns': len(all_campaigns),
    'balance_campaigns': len(balance_campaigns),
    'balance_performance': {
        'sessions': int(total_sessions) if 'total_sessions' in locals() else 0,
        'conversions': int(total_conversions) if 'total_conversions' in locals() else 0,
        'cvr': float(overall_cvr) if 'overall_cvr' in locals() else 0,
        'aov': float(overall_aov) if 'overall_aov' in locals() else 0,
        'revenue': float(total_revenue) if 'total_revenue' in locals() else 0
    }
}

with open(OUTPUT_DIR / "balance_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Analysis saved to {OUTPUT_DIR}")