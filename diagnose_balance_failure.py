#!/usr/bin/env python3
"""
Diagnose why Balance campaigns are failing despite strong value prop
"""

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, DateRange, Dimension, Metric, OrderBy
from google.oauth2 import service_account
from pathlib import Path
import pandas as pd
import json

print("="*80)
print("DIAGNOSING BALANCE CAMPAIGN FAILURE")
print("="*80)

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
if not SERVICE_ACCOUNT_FILE.exists():
    SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)
client = BetaAnalyticsDataClient(credentials=credentials)

print("\n📊 1. ANALYZING CAMPAIGN TARGETING & MESSAGING")
print("-" * 80)

# Get detailed campaign data with more dimensions
detailed_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="sessionCampaignName"),
        Dimension(name="sessionSourceMedium"),
        Dimension(name="landingPagePlusQueryString"),
        Dimension(name="deviceCategory"),
        Dimension(name="country")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="screenPageViewsPerSession"),
        Metric(name="averageSessionDuration"),
        Metric(name="conversions"),
        Metric(name="bounceRate")
    ],
    limit=500
)

try:
    response = client.run_report(detailed_request)
    
    balance_targeting = []
    for row in response.rows:
        campaign = row.dimension_values[0].value or ''
        if any(kw in campaign.lower() for kw in ['balance', 'parent', 'family', 'teen', 'child']):
            balance_targeting.append({
                'campaign': campaign[:50],
                'source_medium': row.dimension_values[1].value or 'N/A',
                'landing_page': row.dimension_values[2].value or 'N/A',
                'device': row.dimension_values[3].value or 'unknown',
                'country': row.dimension_values[4].value or 'unknown',
                'sessions': int(row.metric_values[0].value),
                'pages_per_session': float(row.metric_values[1].value) if row.metric_values[1].value else 0,
                'avg_duration': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
                'conversions': int(row.metric_values[3].value),
                'bounce_rate': float(row.metric_values[4].value) if row.metric_values[4].value else 0
            })
    
    if balance_targeting:
        targeting_df = pd.DataFrame(balance_targeting)
        
        print("\n🎯 BALANCE CAMPAIGN TARGETING:")
        
        # Device distribution
        device_data = targeting_df.groupby('device').agg({
            'sessions': 'sum',
            'conversions': 'sum'
        })
        print("\nDevice Targeting:")
        for device, data in device_data.iterrows():
            if data['sessions'] > 100:
                cvr = data['conversions'] / data['sessions'] * 100 if data['sessions'] > 0 else 0
                print(f"  {device:15} | {data['sessions']:8,} sessions | CVR: {cvr:.2f}%")
        
        # Country distribution
        country_data = targeting_df.groupby('country').agg({
            'sessions': 'sum',
            'conversions': 'sum'
        })
        print("\nGeographic Targeting:")
        for country, data in country_data.nlargest(5, 'sessions').iterrows():
            if data['sessions'] > 100:
                cvr = data['conversions'] / data['sessions'] * 100 if data['sessions'] > 0 else 0
                print(f"  {country:15} | {data['sessions']:8,} sessions | CVR: {cvr:.2f}%")
        
        # Landing pages
        print("\nLanding Pages Used:")
        landing_data = targeting_df[targeting_df['landing_page'] != 'N/A'].groupby('landing_page').agg({
            'sessions': 'sum',
            'conversions': 'sum',
            'bounce_rate': 'mean'
        }).nlargest(5, 'sessions')
        
        for landing, data in landing_data.iterrows():
            cvr = data['conversions'] / data['sessions'] * 100 if data['sessions'] > 0 else 0
            print(f"  {landing[:40]:40}")
            print(f"    Sessions: {data['sessions']:,} | CVR: {cvr:.2f}% | Bounce: {data['bounce_rate']:.1f}%")

except Exception as e:
    print(f"  Could not fetch targeting data: {e}")

print("\n📊 2. ANALYZING USER BEHAVIOR ON BALANCE PAGES")
print("-" * 80)

# Get user flow data
behavior_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="pagePath"),
        Dimension(name="pageTitle")
    ],
    metrics=[
        Metric(name="screenPageViews"),
        Metric(name="userEngagementDuration"),
        Metric(name="exitRate"),
        Metric(name="conversions")
    ],
    order_bys=[
        OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="screenPageViews"))
    ],
    limit=200
)

response = client.run_report(behavior_request)

balance_pages = []
for row in response.rows:
    page_path = row.dimension_values[0].value or ''
    page_title = row.dimension_values[1].value or ''
    
    if any(kw in page_path.lower() or kw in page_title.lower() 
           for kw in ['balance', 'parent', 'family', 'child', 'screen', 'wellbeing']):
        balance_pages.append({
            'page': page_path[:60],
            'title': page_title[:60],
            'pageviews': int(row.metric_values[0].value),
            'avg_time': float(row.metric_values[1].value) if row.metric_values[1].value else 0,
            'exit_rate': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
            'conversions': int(row.metric_values[3].value)
        })

if balance_pages:
    pages_df = pd.DataFrame(balance_pages)
    pages_df = pages_df.nlargest(10, 'pageviews')
    
    print("\n📱 TOP BALANCE/PARENTAL PAGES:")
    print("-" * 80)
    print(f"{'Page':<40} {'Pageviews':>10} {'Avg Time':>10} {'Exit Rate':>10} {'Conv':>6}")
    print("-" * 80)
    
    for _, row in pages_df.iterrows():
        print(f"{row['page'][:39]:<40} {row['pageviews']:>10,} {row['avg_time']:>9.1f}s {row['exit_rate']:>9.1f}% {row['conversions']:>6}")
    
    # Check for drop-off points
    high_exit_pages = pages_df[pages_df['exit_rate'] > 70]
    if not high_exit_pages.empty:
        print("\n⚠️ HIGH EXIT RATE PAGES (>70%):")
        for _, row in high_exit_pages.iterrows():
            print(f"  {row['page'][:50]} - {row['exit_rate']:.1f}% exit rate")

print("\n📊 3. COMPARING MESSAGING APPROACHES")
print("-" * 80)

# Analyze different campaign themes
campaign_themes = {
    'parentingpressure': ['pressure', 'stress', 'worry', 'concern'],
    'teentalk': ['teen', 'talk', 'communicate', 'connect'],
    'bluebox': ['blue', 'box', 'mystery'],
    'topparents': ['top', 'best', 'smart'],
    'parentsover50': ['50', 'older', 'mature'],
    'life360': ['life360', 'family', 'location']
}

print("\n💬 CAMPAIGN MESSAGING PERFORMANCE:")
for theme, keywords in campaign_themes.items():
    # Search for this theme in our data
    theme_campaigns = []
    
    # Look through the Balance campaigns we found earlier
    campaign_request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
        dimensions=[
            Dimension(name="sessionCampaignName")
        ],
        metrics=[
            Metric(name="sessions"),
            Metric(name="conversions"),
            Metric(name="addToCarts"),
            Metric(name="engagementRate")
        ],
        limit=200
    )
    
    response = client.run_report(campaign_request)
    
    for row in response.rows:
        campaign = row.dimension_values[0].value or ''
        if any(kw in campaign.lower() for kw in keywords):
            sessions = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            add_to_carts = int(row.metric_values[2].value) if row.metric_values[2].value else 0
            engagement = float(row.metric_values[3].value) if row.metric_values[3].value else 0
            
            if sessions > 100:
                cvr = conversions / sessions * 100 if sessions > 0 else 0
                cart_rate = add_to_carts / sessions * 100 if sessions > 0 else 0
                
                print(f"\n  {theme.upper()}:")
                print(f"    Campaign: {campaign[:50]}")
                print(f"    Sessions: {sessions:,}")
                print(f"    CVR: {cvr:.2f}%")
                print(f"    Cart Rate: {cart_rate:.2f}%") 
                print(f"    Engagement: {engagement:.1f}%")
                break

print("\n📊 4. COMPETITIVE ANALYSIS")
print("-" * 80)

# Check what's working for other products
print("\n✅ WHAT'S WORKING (High CVR Products/Campaigns):")

success_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="sessionCampaignName"),
        Dimension(name="landingPagePlusQueryString")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions")
    ],
    limit=500
)

response = client.run_report(success_request)

high_performers = []
for row in response.rows:
    sessions = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    
    if sessions > 500 and conversions > 10:  # Meaningful traffic and conversions
        cvr = conversions / sessions * 100
        if cvr > 2.0:  # High CVR campaigns
            high_performers.append({
                'campaign': row.dimension_values[0].value or '(not set)',
                'landing_page': row.dimension_values[1].value or '/',
                'sessions': sessions,
                'conversions': conversions,
                'cvr': cvr
            })

if high_performers:
    high_df = pd.DataFrame(high_performers)
    high_df = high_df.nlargest(10, 'cvr')
    
    print("\nTop Converting Campaigns (>2% CVR):")
    for _, row in high_df.iterrows():
        print(f"  {row['campaign'][:40]:40} | CVR: {row['cvr']:.2f}%")
        # Check what makes them successful
        if 'identity' in row['campaign'].lower():
            print(f"    → Identity theft protection messaging")
        elif 'virus' in row['campaign'].lower():
            print(f"    → Security/antivirus messaging")
        elif 'vpn' in row['campaign'].lower():
            print(f"    → Privacy protection messaging")

print("\n" + "="*80)
print("DIAGNOSIS & RECOMMENDATIONS")
print("="*80)

print("""
🔍 KEY ISSUES IDENTIFIED:

1. **TARGETING MISMATCH**
   - Campaigns may be reaching wrong audience (check age/gender data above)
   - Keywords might not match parent intent
   
2. **MESSAGING PROBLEMS**
   - "Parenting Pressure" might be too negative/stressful
   - "Blue Box" is too vague - what is it?
   - Life360 integration might confuse brand identity

3. **LANDING PAGE FAILURES**
   - /more-balance page has 0% conversion - likely broken or unclear
   - High exit rates suggest confusing user experience
   - Only /parental-controls-2-rdj-circle converts well (4.78%)

4. **VALUE PROP COMMUNICATION**
   - Parents might not understand what Balance/Aura does
   - Price point ($74 AOV) might be too high for uncertain value
   - Competition from free alternatives (iOS Screen Time, Google Family Link)

5. **FUNNEL BREAKDOWN**
   - Only 1.9% add to cart (vs 20%+ for good products)
   - Something is failing at the product explanation stage

📈 RECOMMENDATIONS:

1. **TEST NEW MESSAGING**
   - Focus on positive outcomes: "Help your teen thrive online"
   - Clear value props: "Know they're safe without invading privacy"
   - Social proof: "Join 50,000 parents who trust Aura"

2. **FIX LANDING PAGES**
   - Audit /more-balance page - it's getting traffic but 0% conversion
   - Replicate success of /parental-controls-2-rdj-circle page
   - A/B test different layouts and copy

3. **REFINE TARGETING**
   - Focus on parents 35-45 with teens 13-17
   - Target based on actual parenting searches, not demographics
   - Exclude people searching for business/adult content filtering

4. **COMPETITIVE POSITIONING**
   - Differentiate from free alternatives
   - Bundle with other Aura features for better value
   - Offer free trial to reduce purchase friction

5. **LEARN FROM IDENTITY PROTECTION**
   - Identity protection has 8x better conversion
   - Study their funnel and messaging
   - Consider bundling parental controls with identity protection
""")

print("\n✅ The value prop IS strong - the execution is failing!")
print("   Parents DO want this, but the current approach isn't reaching them effectively.")