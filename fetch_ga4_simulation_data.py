#!/usr/bin/env python3
"""
Fetch specific GA4 data to enhance GAELP simulation realism
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, DateRange, Dimension, Metric, OrderBy
)
from google.oauth2 import service_account

print("="*70)
print("FETCHING GA4 DATA FOR ULTRA-REALISTIC SIMULATION")
print("="*70)

# Configuration
GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
OUTPUT_DIR = Path("/home/hariravichandran/AELP/data/ga4_simulation_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Check for service account file
if not SERVICE_ACCOUNT_FILE.exists():
    alt_path = Path.home() / '.config' / 'gaelp' / 'service-account.json'
    if alt_path.exists():
        SERVICE_ACCOUNT_FILE = alt_path

print(f"\nUsing service account: {SERVICE_ACCOUNT_FILE}")

# Create credentials
credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

# 1. HOURLY PATTERNS FOR BID PACING
print("\n📊 1. Fetching hourly patterns for bid pacing...")
hourly_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="hour"),
        Dimension(name="dayOfWeek")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue"),
        Metric(name="engagementRate")
    ]
)

hourly_response = client.run_report(hourly_request)
hourly_data = []
for row in hourly_response.rows:
    hourly_data.append({
        'hour': int(row.dimension_values[0].value),
        'day_of_week': int(row.dimension_values[1].value),
        'sessions': int(row.metric_values[0].value),
        'conversions': int(row.metric_values[1].value),
        'revenue': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
        'engagement_rate': float(row.metric_values[3].value) if row.metric_values[3].value else 0
    })

hourly_df = pd.DataFrame(hourly_data)
hourly_df.to_csv(OUTPUT_DIR / "hourly_patterns.csv", index=False)
print(f"   ✅ Saved {len(hourly_df)} hourly pattern records")

# Calculate peak hours
peak_hours = hourly_df.groupby('hour')['conversions'].sum().nlargest(5).index.tolist()
print(f"   📈 Peak conversion hours: {peak_hours}")

# 2. CHANNEL-SPECIFIC CONVERSION DATA
print("\n📊 2. Fetching channel-specific conversion rates...")
channel_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="sessionDefaultChannelGroup"),
        Dimension(name="deviceCategory")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue"),
        Metric(name="averageSessionDuration"),
        Metric(name="bounceRate"),
        Metric(name="screenPageViewsPerSession")
    ]
)

channel_response = client.run_report(channel_request)
channel_data = []
for row in channel_response.rows:
    channel_data.append({
        'channel': row.dimension_values[0].value or 'Direct',
        'device': row.dimension_values[1].value or 'desktop',
        'sessions': int(row.metric_values[0].value),
        'conversions': int(row.metric_values[1].value),
        'revenue': float(row.metric_values[2].value) if row.metric_values[2].value else 0,
        'avg_duration': float(row.metric_values[3].value) if row.metric_values[3].value else 0,
        'bounce_rate': float(row.metric_values[4].value) if row.metric_values[4].value else 0,
        'pages_per_session': float(row.metric_values[5].value) if row.metric_values[5].value else 1
    })

channel_df = pd.DataFrame(channel_data)
channel_df['cvr'] = channel_df['conversions'] / channel_df['sessions'].replace(0, 1)
channel_df['avg_order_value'] = channel_df['revenue'] / channel_df['conversions'].replace(0, 1)
channel_df.to_csv(OUTPUT_DIR / "channel_performance.csv", index=False)
print(f"   ✅ Saved {len(channel_df)} channel performance records")

# Show top channels by CVR
top_channels = channel_df.nlargest(5, 'cvr')[['channel', 'cvr', 'avg_order_value']]
print("\n   Top channels by conversion rate:")
for _, row in top_channels.iterrows():
    print(f"   - {row['channel']:20} CVR: {row['cvr']*100:.2f}%  AOV: ${row['avg_order_value']:.2f}")

# 3. USER JOURNEY LENGTH (MULTI-TOUCH ATTRIBUTION)
print("\n📊 3. Fetching user journey data for attribution...")
journey_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="30daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="sessionDefaultChannelGroup")
    ],
    metrics=[
        Metric(name="sessionsPerUser"),
        Metric(name="userEngagementDuration"),
        Metric(name="totalUsers"),
        Metric(name="newUsers")
    ]
)

journey_response = client.run_report(journey_request)
journey_data = []
for row in journey_response.rows:
    journey_data.append({
        'channel': row.dimension_values[0].value or 'Direct',
        'sessions_per_user': float(row.metric_values[0].value) if row.metric_values[0].value else 1,
        'engagement_duration': float(row.metric_values[1].value) if row.metric_values[1].value else 0,
        'total_users': int(row.metric_values[2].value),
        'new_users': int(row.metric_values[3].value)
    })

journey_df = pd.DataFrame(journey_data)
journey_df['returning_user_rate'] = 1 - (journey_df['new_users'] / journey_df['total_users'].replace(0, 1))
journey_df.to_csv(OUTPUT_DIR / "user_journeys.csv", index=False)
print(f"   ✅ Saved {len(journey_df)} user journey records")
print(f"   📊 Average sessions before conversion: {journey_df['sessions_per_user'].mean():.2f}")

# 4. GEOGRAPHIC PERFORMANCE
print("\n📊 4. Fetching geographic data for user personas...")
geo_request = RunReportRequest(
    property=f"properties/{GA_PROPERTY_ID}",
    date_ranges=[DateRange(start_date="7daysAgo", end_date="yesterday")],
    dimensions=[
        Dimension(name="country"),
        Dimension(name="region")
    ],
    metrics=[
        Metric(name="sessions"),
        Metric(name="conversions"),
        Metric(name="purchaseRevenue")
    ],
    limit=50  # Top 50 regions
)

geo_response = client.run_report(geo_request)
geo_data = []
for row in geo_response.rows:
    geo_data.append({
        'country': row.dimension_values[0].value,
        'region': row.dimension_values[1].value or 'Unknown',
        'sessions': int(row.metric_values[0].value),
        'conversions': int(row.metric_values[1].value),
        'revenue': float(row.metric_values[2].value) if row.metric_values[2].value else 0
    })

geo_df = pd.DataFrame(geo_data)
geo_df['cvr'] = geo_df['conversions'] / geo_df['sessions'].replace(0, 1)
geo_df.to_csv(OUTPUT_DIR / "geographic_data.csv", index=False)
print(f"   ✅ Saved {len(geo_df)} geographic records")

# 5. CALCULATE KEY SIMULATION PARAMETERS
print("\n" + "="*70)
print("SIMULATION PARAMETERS FROM REAL DATA")
print("="*70)

simulation_params = {
    "peak_hours": peak_hours,
    "avg_sessions_per_user": journey_df['sessions_per_user'].mean(),
    "channel_cvr": channel_df.groupby('channel')['cvr'].mean().to_dict(),
    "device_distribution": channel_df.groupby('device')['sessions'].sum().to_dict(),
    "avg_order_value": channel_df['avg_order_value'].mean(),
    "bounce_rate_by_channel": channel_df.groupby('channel')['bounce_rate'].mean().to_dict(),
    "pages_per_session": channel_df['pages_per_session'].mean(),
    "returning_user_rate": journey_df['returning_user_rate'].mean()
}

# Save simulation parameters
params_file = OUTPUT_DIR / "simulation_parameters.json"
with open(params_file, 'w') as f:
    json.dump(simulation_params, f, indent=2, default=str)

print(f"\n✅ Saved simulation parameters to {params_file}")
print("\n📊 Key Parameters:")
print(f"   - Peak hours: {simulation_params['peak_hours']}")
print(f"   - Avg sessions/user: {simulation_params['avg_sessions_per_user']:.2f}")
print(f"   - Avg order value: ${simulation_params['avg_order_value']:.2f}")
print(f"   - Pages per session: {simulation_params['pages_per_session']:.2f}")
print(f"   - Returning user rate: {simulation_params['returning_user_rate']*100:.1f}%")

print("\n🎯 Top 3 channels by CVR:")
sorted_channels = sorted(simulation_params['channel_cvr'].items(), key=lambda x: x[1], reverse=True)[:3]
for channel, cvr in sorted_channels:
    print(f"   - {channel:20} {cvr*100:.2f}%")

print("\n✨ This data will make GAELP simulation incredibly realistic!")
print("   - Real conversion patterns by hour and channel")
print("   - Actual user journey lengths for attribution")
print("   - True geographic and device distributions")
print("   - Genuine engagement and bounce rates")