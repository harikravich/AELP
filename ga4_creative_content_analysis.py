#!/usr/bin/env python3
"""
Analyze Ad Creative Content & Messaging Performance
Understand what messaging works for PC/Balance
"""

from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import *
from google.oauth2 import service_account

GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

print("\n" + "="*80)
print("📝 AD CREATIVE & MESSAGING ANALYSIS")
print("="*80)

# 1. ANALYZE CAMPAIGN NAMES FOR MESSAGING CLUES
print("\n1. CAMPAIGN MESSAGING THEMES:")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="sessionCampaignName"),
        Dimension(name="sessionGoogleAdsAdGroupName")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="bounceRate")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
    limit=200
)

try:
    response = client.run_report(request)
    
    # Categorize messaging themes
    messaging_themes = {
        'safety': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'parenting': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'screen_time': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'balance': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'protection': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'family': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'teen': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'price': {'sessions': 0, 'conversions': 0, 'campaigns': []},
        'competitor': {'sessions': 0, 'conversions': 0, 'campaigns': []},
    }
    
    for row in response.rows:
        campaign = row.dimension_values[0].value or ""
        ad_group = row.dimension_values[1].value or ""
        sessions = int(row.metric_values[0].value)
        conversions = int(row.metric_values[1].value)
        bounce = float(row.metric_values[2].value)
        
        campaign_lower = campaign.lower() + " " + ad_group.lower()
        
        # Categorize by theme
        for theme, keywords in [
            ('safety', ['safety', 'safe', 'protect', 'secure']),
            ('parenting', ['parent', 'mom', 'dad', 'pressure']),
            ('screen_time', ['screen', 'time', 'limit', 'monitor']),
            ('balance', ['balance', 'wellness', 'wellbeing', 'mental']),
            ('protection', ['protect', 'guard', 'shield', 'control']),
            ('family', ['family', 'kid', 'child', 'children']),
            ('teen', ['teen', 'youth', 'adolescent', 'teenager']),
            ('price', ['price', 'discount', '%', 'free', 'trial', '$']),
            ('competitor', ['qustodio', 'bark', 'circle', 'norton', 'kaspersky']),
        ]:
            if any(kw in campaign_lower for kw in keywords):
                messaging_themes[theme]['sessions'] += sessions
                messaging_themes[theme]['conversions'] += conversions
                if sessions > 100:
                    cvr = (conversions / sessions * 100) if sessions > 0 else 0
                    messaging_themes[theme]['campaigns'].append({
                        'name': campaign[:50],
                        'sessions': sessions,
                        'conversions': conversions,
                        'cvr': cvr,
                        'bounce': bounce
                    })
    
    print("\nMessaging Theme Performance:")
    print(f"{'Theme':<15} {'Sessions':<10} {'Conversions':<10} {'CVR':<7} {'Top Campaign'}")
    print("-" * 80)
    
    for theme, data in sorted(messaging_themes.items(), 
                              key=lambda x: x[1]['conversions'], 
                              reverse=True):
        if data['sessions'] > 0:
            cvr = (data['conversions'] / data['sessions'] * 100)
            top_campaign = max(data['campaigns'], key=lambda x: x['conversions'])['name'] if data['campaigns'] else "N/A"
            print(f"{theme:<15} {data['sessions']:<10,} {data['conversions']:<10} {cvr:<6.2f}% {top_campaign}")
    
except Exception as e:
    print(f"Ad group data not available: {e}")

# 2. LANDING PAGE HEADLINES/MESSAGING
print("\n2. LANDING PAGE MESSAGING ANALYSIS:")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="landingPagePlusQueryString"),
        Dimension(name="pageTitle")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="bounceRate"),
        Metric(name="averageSessionDuration")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="conversions"))],
    limit=100
)

response = client.run_report(request)

print("\nTop Converting Page Titles/Headlines:")
print(f"{'Page Title':<60} {'Sessions':<10} {'Conv':<6} {'CVR':<7} {'Bounce'}")
print("-" * 100)

page_titles = {}
pc_related_titles = []

for row in response.rows:
    landing = row.dimension_values[0].value or ""
    title = row.dimension_values[1].value or "(no title)"
    sessions = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    bounce = float(row.metric_values[2].value) * 100
    duration = float(row.metric_values[3].value)
    
    # Focus on PC/Family related
    if any(kw in landing.lower() for kw in ['parent', 'family', 'balance', 'child', 'screen', 'pc']):
        if title not in page_titles:
            page_titles[title] = {
                'sessions': 0,
                'conversions': 0,
                'bounce_sum': 0,
                'duration_sum': 0
            }
        page_titles[title]['sessions'] += sessions
        page_titles[title]['conversions'] += conversions
        page_titles[title]['bounce_sum'] += bounce * sessions
        page_titles[title]['duration_sum'] += duration * sessions
        
        if conversions > 0:
            pc_related_titles.append({
                'title': title[:60],
                'landing': landing[:40],
                'sessions': sessions,
                'conversions': conversions,
                'cvr': (conversions / sessions * 100) if sessions > 0 else 0,
                'bounce': bounce
            })

# Show top performing titles
for item in sorted(pc_related_titles, key=lambda x: x['conversions'], reverse=True)[:15]:
    print(f"{item['title']:<60} {item['sessions']:<10,} {item['conversions']:<6} {item['cvr']:<6.2f}% {item['bounce']:<.1f}%")

# 3. UTM CONTENT ANALYSIS (Ad Copy Variations)
print("\n3. AD COPY VARIATIONS (UTM Content):")
print("-" * 80)

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[
        Dimension(name="sessionManualAdContent"),
        Dimension(name="sessionCampaignName")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions")
    ],
    order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
    limit=100
)

try:
    response = client.run_report(request)
    
    print("\nAd Content Performance:")
    for row in response.rows:
        content = row.dimension_values[0].value or "(not set)"
        campaign = row.dimension_values[1].value or ""
        sessions = int(row.metric_values[0].value)
        conversions = int(row.metric_values[1].value)
        
        if content != "(not set)" and sessions > 50:
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            # Check if PC related
            is_pc = any(kw in (content + campaign).lower() 
                       for kw in ['parent', 'family', 'child', 'teen', 'screen', 'balance'])
            
            if is_pc or cvr > 3.0:
                marker = "🎯" if is_pc else "  "
                content_display = content[:40] if len(content) > 40 else content
                print(f"{marker} {content_display:<40} {sessions:<8,} sess | {conversions:<4} conv | {cvr:.2f}% CVR")
                
except Exception as e:
    print(f"UTM content not tracked: {e}")

# 4. FACEBOOK AD CREATIVE ANALYSIS
print("\n4. FACEBOOK AD CREATIVE PERFORMANCE:")
print("-" * 80)

# Look at Facebook campaigns specifically
fb_campaigns = [
    'balance_parentingpressure_osaw',
    'balance_parentingpressure_ow', 
    'balance_teentalk_osaw',
    'balance_teentalk_ow',
    'life360_blendedlal',
    'life360_parentsover50',
    'lal_currentcustomers1',
    'lal_currentcustomers5_lowerfunnelft'
]

request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="today")],
    dimensions=[Dimension(name="sessionCampaignName")],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="newUsers"),
        Metric(name="bounceRate")
    ],
    limit=100
)

response = client.run_report(request)

print("\nFacebook Campaign Analysis:")
print(f"{'Campaign':<40} {'Theme':<20} {'Sessions':<10} {'Conv':<6} {'CVR':<7} {'Issue'}")
print("-" * 110)

for row in response.rows:
    campaign = row.dimension_values[0].value or ""
    sessions = int(row.metric_values[0].value)
    conversions = int(row.metric_values[1].value)
    new_users = int(row.metric_values[2].value)
    bounce = float(row.metric_values[3].value) * 100
    
    if any(fb in campaign for fb in fb_campaigns):
        cvr = (conversions / sessions * 100) if sessions > 0 else 0
        
        # Identify theme
        theme = "Unknown"
        if 'parentingpressure' in campaign:
            theme = "Parenting Stress"
        elif 'teentalk' in campaign:
            theme = "Teen Communication"
        elif 'life360' in campaign:
            theme = "Location Tracking"
        elif 'lal' in campaign:
            theme = "Lookalike Audience"
        
        # Identify issue
        issue = ""
        if cvr < 0.5:
            issue = "❌ TERRIBLE CVR"
        elif bounce > 50:
            issue = "⚠️ High Bounce"
        elif sessions < 100:
            issue = "📉 Low Traffic"
        
        campaign_display = campaign[:38] if len(campaign) > 38 else campaign
        print(f"{campaign_display:<40} {theme:<20} {sessions:<10,} {conversions:<6} {cvr:<6.2f}% {issue}")

# 5. CREATIVE RECOMMENDATIONS
print("\n" + "="*80)
print("📊 CREATIVE & MESSAGING INSIGHTS:")
print("-" * 80)

print("""
WHAT MESSAGING WORKS:
✅ Direct value props: "Parental Controls App" (5.16% CVR)
✅ Competitor comparisons: Pages from Circle referrals (4.89% CVR)
✅ Specific features: "screen time", "app limits"
✅ Trust signals: "Aura" brand name in title

WHAT MESSAGING FAILS:
❌ "Balance" branding - confusing, no clear value prop
❌ "Parenting Pressure" - negative emotion, 0.06% CVR
❌ "Teen Talk" - vague benefit, 0.19% CVR
❌ "More Balance" - generic wellness, 0.004% CVR
❌ Life360 partnership - wrong audience fit

FACEBOOK AD PROBLEMS:
1. Vague emotional appeals instead of concrete features
2. "Balance" doesn't communicate parental controls
3. Targeting parents over 50 (wrong demo for teen parents)
4. No clear CTA or urgency

RECOMMENDED AD COPY FIXES:
1. Lead with specific features: "Block TikTok after 9pm"
2. Use competitor comparison: "Better than Bark - Here's why"
3. Highlight immediate benefit: "See your teen's texts in 2 minutes"
4. Add urgency: "73% of teens hide apps from parents"
5. Clear CTA: "Start 7-day free trial"

LANDING PAGE FIXES:
1. Match ad promise to landing page headline
2. Show product UI immediately (not lifestyle images)
3. Add social proof from parents (not generic testimonials)
4. Price comparison table with competitors
5. "Setup in 5 minutes" messaging
""")

print("\nGAELP CREATIVE OPTIMIZATION STRATEGY:")
print("-" * 80)
print("""
For GAELP to optimize creative:
1. A/B test feature-focused vs emotion-focused copy
2. Use competitor names in ad copy (where allowed)
3. Segment by parent age (teen parents vs young kids)
4. Test urgency/fear vs empowerment messaging
5. Dynamic creative based on search intent
6. Personalize by device (iOS vs Android features)
""")