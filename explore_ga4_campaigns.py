#!/usr/bin/env python3
"""
Explore GA4 campaigns and Parental Controls specific data
"""

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, DateRange, Dimension, Metric, OrderBy
)
from google.oauth2 import service_account
from pathlib import Path
import pandas as pd
import json

print("="*70)
print("EXPLORING GA4 CAMPAIGNS AND PARENTAL CONTROLS DATA")
print("="*70)

# Configuration
GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
OUTPUT_DIR = Path("/home/hariravichandran/AELP/data/ga4_campaign_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

if not SERVICE_ACCOUNT_FILE.exists():
    SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

# 1. GET TOP CAMPAIGNS
print("\n📊 1. Fetching top campaigns...")
campaign_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="sessionCampaignName"),
        Dimension(name="sessionSource"),
        Dimension(name="sessionMedium")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue"),
        Metric(name="engagementRate")
    ],
    order_bys=[
        OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))
    ],
    limit=50  # Top 50 campaigns
)

try:
    campaign_response = client.run_report(campaign_request)
    campaigns = []
    for row in campaign_response.rows:
        campaigns.append({
            'campaign': row.dimension_values[0].value or '(not set)',
            'source': row.dimension_values[1].value or '(direct)',
            'medium': row.dimension_values[2].value or '(none)',
            'sessions': int(row.metric_values[0].value),
            'conversions': int(row.metric_values[1].value),
            'revenue': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
            'engagement_rate': float(row.metric_values[3].value) if row.metric_values[3].value else 0
        })
    
    campaigns_df = pd.DataFrame(campaigns)
    campaigns_df['cvr'] = campaigns_df['conversions'] / campaigns_df['sessions'].replace(0, 1)
    campaigns_df.to_csv(OUTPUT_DIR / "top_campaigns.csv", index=False)
    
    print(f"✅ Found {len(campaigns_df)} campaigns")
    
    # Show top campaigns
    print("\n📈 Top 5 Campaigns by Sessions:")
    for i, row in campaigns_df.head(5).iterrows():
        print(f"   {row['campaign'][:40]:40} | {row['sessions']:,} sessions | CVR: {row['cvr']*100:.2f}%")
    
    # Look for parental controls related campaigns
    parental_keywords = ['parent', 'family', 'child', 'kid', 'safe', 'control', 'monitor', 'screen']
    parental_campaigns = campaigns_df[
        campaigns_df['campaign'].str.lower().str.contains('|'.join(parental_keywords), na=False)
    ]
    
    if not parental_campaigns.empty:
        print(f"\n👨‍👩‍👧‍👦 Found {len(parental_campaigns)} Parental Controls related campaigns:")
        for _, row in parental_campaigns.head(5).iterrows():
            print(f"   {row['campaign'][:40]:40} | CVR: {row['cvr']*100:.2f}%")
    
except Exception as e:
    print(f"Error fetching campaigns: {e}")

# 2. GET LANDING PAGE DATA
print("\n📊 2. Fetching landing page performance...")
landing_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="landingPagePlusQueryString"),
        Dimension(name="sessionDefaultChannelGroup")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="bounceRate"),
        Metric(name="averageSessionDuration")
    ],
    order_bys=[
        OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))
    ],
    limit=30
)

try:
    landing_response = client.run_report(landing_request)
    landing_pages = []
    for row in landing_response.rows:
        landing_pages.append({
            'landing_page': row.dimension_values[0].value or '/',
            'channel': row.dimension_values[1].value or 'Direct',
            'sessions': int(row.metric_values[0].value),
            'conversions': int(row.metric_values[1].value),
            'bounce_rate': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
            'avg_duration': float(row.metric_values[3].value) if row.metric_values[3].value else 0
        })
    
    landing_df = pd.DataFrame(landing_pages)
    landing_df['cvr'] = landing_df['conversions'] / landing_df['sessions'].replace(0, 1)
    landing_df.to_csv(OUTPUT_DIR / "landing_pages.csv", index=False)
    
    print(f"✅ Analyzed {len(landing_df)} landing pages")
    
    # Look for feature-specific pages
    feature_pages = {
        'parental_controls': ['parent', 'family', 'child', 'screen-time', 'monitor'],
        'location': ['location', 'find', 'track', 'gps'],
        'vpn': ['vpn', 'privacy', 'secure'],
        'antivirus': ['virus', 'malware', 'protect'],
        'identity': ['identity', 'theft', 'credit']
    }
    
    print("\n🎯 Feature-Specific Landing Pages:")
    for feature, keywords in feature_pages.items():
        feature_landings = landing_df[
            landing_df['landing_page'].str.lower().str.contains('|'.join(keywords), na=False)
        ]
        if not feature_landings.empty:
            total_sessions = feature_landings['sessions'].sum()
            avg_cvr = feature_landings['cvr'].mean()
            print(f"   {feature:20} | {total_sessions:,} sessions | CVR: {avg_cvr*100:.2f}%")
    
except Exception as e:
    print(f"Error fetching landing pages: {e}")

# 3. GET CUSTOM EVENTS (looking for feature usage)
print("\n📊 3. Fetching custom events...")
events_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="yesterday", end_date="yesterday")],
    dimensions=[
        Dimension(name="eventName")
    ],
    metrics=[
        Metric(name="eventCount"),
        Metric(name="totalUsers")
    ],
    order_bys=[
        OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))
    ],
    limit=100
)

try:
    events_response = client.run_report(events_request)
    events = []
    for row in events_response.rows:
        events.append({
            'event': row.dimension_values[0].value,
            'count': int(row.metric_values[0].value),
            'users': int(row.metric_values[1].value)
        })
    
    events_df = pd.DataFrame(events)
    events_df.to_csv(OUTPUT_DIR / "custom_events.csv", index=False)
    
    print(f"✅ Found {len(events_df)} event types")
    
    # Show key events
    key_events = ['purchase', 'sign_up', 'trial_start', 'add_to_cart', 'begin_checkout']
    print("\n🔑 Key Conversion Events:")
    for event in key_events:
        event_data = events_df[events_df['event'] == event]
        if not event_data.empty:
            count = event_data['count'].values[0]
            users = event_data['users'].values[0]
            print(f"   {event:20} | {count:,} events | {users:,} users")
    
    # Look for feature-specific events
    print("\n🎯 Feature-Related Events:")
    feature_keywords = {
        'parental': ['parent', 'child', 'family', 'screen', 'limit', 'block'],
        'location': ['location', 'gps', 'find', 'track'],
        'security': ['vpn', 'virus', 'malware', 'secure', 'protect']
    }
    
    for feature, keywords in feature_keywords.items():
        feature_events = events_df[
            events_df['event'].str.lower().str.contains('|'.join(keywords), na=False)
        ]
        if not feature_events.empty:
            total_events = feature_events['count'].sum()
            print(f"   {feature:20} | {total_events:,} events")
    
except Exception as e:
    print(f"Error fetching events: {e}")

# 4. GET USER PROPERTIES/AUDIENCES
print("\n📊 4. Fetching audience segments...")
audience_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="newVsReturning"),
        Dimension(name="deviceCategory"),
        Dimension(name="operatingSystem")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue")
    ]
)

try:
    audience_response = client.run_report(audience_request)
    audiences = []
    for row in audience_response.rows:
        audiences.append({
            'user_type': row.dimension_values[0].value,
            'device': row.dimension_values[1].value,
            'os': row.dimension_values[2].value,
            'sessions': int(row.metric_values[0].value),
            'conversions': int(row.metric_values[1].value),
            'revenue': float(row.metric_values[2].value) if row.metric_values[2].value else 0
        })
    
    audience_df = pd.DataFrame(audiences)
    audience_df['cvr'] = audience_df['conversions'] / audience_df['sessions'].replace(0, 1)
    audience_df.to_csv(OUTPUT_DIR / "audience_segments.csv", index=False)
    
    print(f"✅ Analyzed {len(audience_df)} audience segments")
    
    # iOS is important for Parental Controls
    ios_data = audience_df[audience_df['os'].str.contains('iOS', na=False)]
    if not ios_data.empty:
        ios_sessions = ios_data['sessions'].sum()
        ios_cvr = ios_data['conversions'].sum() / ios_sessions if ios_sessions > 0 else 0
        print(f"\n📱 iOS Performance (Parental Controls primary platform):")
        print(f"   Sessions: {ios_sessions:,}")
        print(f"   CVR: {ios_cvr*100:.2f}%")
    
except Exception as e:
    print(f"Error fetching audiences: {e}")

# 5. SUMMARY AND MAPPING
print("\n" + "="*70)
print("GA4 DATA MAPPING SUMMARY")
print("="*70)

print("""
📋 Available Data for GAELP Enhancement:

1. **Campaign Data**
   - Campaign names, sources, mediums
   - Campaign-specific conversion rates
   - Revenue by campaign
   
2. **Landing Pages**
   - Feature-specific pages (parental controls, vpn, etc.)
   - Conversion rates by landing page
   - Bounce rates and engagement
   
3. **Custom Events**
   - Purchase, sign_up, trial_start
   - Feature usage events
   - User engagement patterns
   
4. **Audience Segments**
   - iOS vs Android (critical for Parental Controls)
   - New vs Returning users
   - Device and OS combinations

🎯 Key Insights for Simulation:
   - iOS users are primary target (Parental Controls iOS-only)
   - Multiple product features affect conversion differently
   - Campaign performance varies significantly
   - Landing page quality affects conversion rates
""")

# Create comprehensive mapping
mapping = {
    "data_available": {
        "campaigns": len(campaigns_df) if 'campaigns_df' in locals() else 0,
        "landing_pages": len(landing_df) if 'landing_df' in locals() else 0,
        "events": len(events_df) if 'events_df' in locals() else 0,
        "segments": len(audience_df) if 'audience_df' in locals() else 0
    },
    "recommendations": [
        "Use campaign-specific CVRs for different ad groups",
        "Model iOS vs Android conversion differently",
        "Include landing page quality in Quality Score",
        "Track multi-product conversion paths",
        "Separate Parental Controls from other features"
    ]
}

with open(OUTPUT_DIR / "ga4_data_mapping.json", 'w') as f:
    json.dump(mapping, f, indent=2)

print(f"\n✅ Data mapping saved to {OUTPUT_DIR}")
print("\n🚀 Ready to build feature-specific conversion models!")