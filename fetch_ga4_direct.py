#!/usr/bin/env python3
"""
Fetch real CTR data directly from GA4 API
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, DateRange, Dimension, Metric, OrderBy
)
from google.oauth2 import service_account

print("="*60)
print("FETCHING REAL CTR DATA FROM GA4 - DIRECT API")
print("="*60)

# Configuration
GA_PROPERTY_ID = "308028264"
SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
OUTPUT_DIR = Path("/home/hariravichandran/AELP/data/ga4_real_ctr")
OUTPUT_DIR.mkdir(exist_ok=True)

# Check for service account file
if not SERVICE_ACCOUNT_FILE.exists():
    alt_path = Path.home() / '.config' / 'gaelp' / 'service-account.json'
    if alt_path.exists():
        SERVICE_ACCOUNT_FILE = alt_path
    else:
        print(f"❌ Service account file not found")
        exit(1)

print(f"Using service account: {SERVICE_ACCOUNT_FILE}")

# Create credentials
credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE),
    scopes=['https://www.googleapis.com/auth/analytics.readonly']
)

client = BetaAnalyticsDataClient(credentials=credentials)

# Fetch data in chunks
START_DATE = datetime(2024, 10, 1)
END_DATE = datetime(2025, 1, 28)
CHUNK_DAYS = 14  # Two-week chunks

all_data = []
current_date = START_DATE

print(f"\nFetching data from {START_DATE.date()} to {END_DATE.date()}")
print(f"Using {CHUNK_DAYS}-day chunks\n")

while current_date < END_DATE:
    chunk_end = min(current_date + timedelta(days=CHUNK_DAYS - 1), END_DATE)
    
    print(f"Fetching: {current_date.date()} to {chunk_end.date()}...", end=" ")
    
    request = RunReportRequest(
        property=f"properties/{GA_PROPERTY_ID}",
        date_ranges=[DateRange(
            start_date=current_date.strftime('%Y-%m-%d'),
            end_date=chunk_end.strftime('%Y-%m-%d')
        )],
        dimensions=[
            Dimension(name="date"),
            Dimension(name="sessionDefaultChannelGroup"),
            Dimension(name="sessionSource"),
            Dimension(name="sessionMedium"),
            Dimension(name="deviceCategory"),
            Dimension(name="hour"),
        ],
        metrics=[
            Metric(name="sessions"),
            Metric(name="totalUsers"),
            Metric(name="screenPageViews"),
            Metric(name="engagementRate"),
            Metric(name="conversions"),
            Metric(name="bounceRate"),
            Metric(name="averageSessionDuration")
        ],
        order_bys=[
            OrderBy(desc=True, dimension=OrderBy.DimensionOrderBy(dimension_name="date"))
        ]
    )
    
    try:
        response = client.run_report(request)
        
        # Convert to dict format
        for row in response.rows:
            row_data = {
                'date': row.dimension_values[0].value,
                'channel': row.dimension_values[1].value or 'direct',
                'source': row.dimension_values[2].value or 'direct',
                'medium': row.dimension_values[3].value or 'none',
                'device': row.dimension_values[4].value or 'desktop',
                'hour': int(row.dimension_values[5].value) if row.dimension_values[5].value else 12,
                'sessions': int(row.metric_values[0].value),
                'users': int(row.metric_values[1].value),
                'pageviews': int(row.metric_values[2].value),
                'engagement_rate': float(row.metric_values[3].value) if row.metric_values[3].value else 0,
                'conversions': int(row.metric_values[4].value),
                'bounce_rate': float(row.metric_values[5].value) if row.metric_values[5].value else 0,
                'avg_session_duration': float(row.metric_values[6].value) if row.metric_values[6].value else 0
            }
            all_data.append(row_data)
        
        print(f"✅ Got {len(response.rows)} rows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    current_date = chunk_end + timedelta(days=1)

print(f"\n\nTotal rows collected: {len(all_data)}")

if all_data:
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save raw data
    raw_file = OUTPUT_DIR / "ga4_raw_data.csv"
    df.to_csv(raw_file, index=False)
    print(f"✅ Saved raw data to {raw_file}")
    
    # Create Criteo-compatible training data
    print("\nConverting to Criteo format for CTR training...")
    
    criteo_data = []
    for _, row in df.iterrows():
        # Determine if this would be a "click" (high engagement)
        # In web analytics, engagement > 50% is like a click in ads
        click = 1 if row['engagement_rate'] > 50 and row['bounce_rate'] < 50 else 0
        
        # Intent signal from channel/source
        intent_score = 1.0
        if row['channel'] == 'Organic Search':
            intent_score = 2.5  # High intent
        elif row['channel'] == 'Paid Search':
            intent_score = 2.0
        elif row['channel'] == 'Direct':
            intent_score = 1.5
        elif row['channel'] == 'Social':
            intent_score = 0.8
        elif row['channel'] == 'Display':
            intent_score = 0.5  # Low intent
        
        # Time of day signal
        hour_signal = 1.0
        if 22 <= row['hour'] or row['hour'] <= 2:  # Late night
            hour_signal = 1.5
        elif 9 <= row['hour'] <= 17:  # Business hours
            hour_signal = 1.2
        
        criteo_row = {
            'click': click,
            
            # Numerical features (observable signals)
            'num_0': intent_score,  # Intent from channel
            'num_1': row['hour'] / 24.0,  # Normalized hour
            'num_2': datetime.strptime(row['date'], '%Y%m%d').weekday() / 7.0,  # Day of week
            'num_3': row['pageviews'] / max(1, row['sessions']),  # Pages per session
            'num_4': row['avg_session_duration'] / 300.0,  # Normalized duration (5 min baseline)
            'num_5': row['engagement_rate'] / 100.0,  # Engagement rate
            'num_6': (100 - row['bounce_rate']) / 100.0,  # Non-bounce rate
            'num_7': 1.0 if row['device'] == 'mobile' else 0.5,  # Device signal
            'num_8': row['conversions'] / max(1, row['sessions']),  # Session CVR
            'num_9': hour_signal,  # Time of day signal
            'num_10': row['sessions'] / 100.0,  # Normalized session volume
            'num_11': row['users'] / 100.0,  # Normalized user volume
            'num_12': int(row['date'][4:6]) / 12.0,  # Month normalized
            
            # Categorical features
            'cat_0': row['channel'],
            'cat_1': row['source'][:20] if row['source'] else '',
            'cat_2': row['medium'],
            'cat_3': row['device'],
            'cat_4': str(row['hour'] // 6),  # Time segment (0-3)
            'cat_5': 'behavioral_health',  # Industry
            'cat_6': row['date'][:6],  # Year-month
            'cat_7': 'aura',  # Brand
            'cat_8': datetime.strptime(row['date'], '%Y%m%d').strftime('%A'),  # Day name
        }
        
        # Fill remaining categorical features
        for i in range(9, 26):
            criteo_row[f'cat_{i}'] = ''
        
        criteo_data.append(criteo_row)
    
    # Save Criteo format
    criteo_df = pd.DataFrame(criteo_data)
    criteo_file = OUTPUT_DIR / "ga4_criteo_training.csv"
    criteo_df.to_csv(criteo_file, index=False)
    print(f"✅ Saved Criteo training data to {criteo_file}")
    
    # Calculate real CTR statistics
    print("\n" + "="*60)
    print("REAL CTR STATISTICS FROM GA4")
    print("="*60)
    
    # Overall stats
    total_impressions = len(criteo_df)
    total_clicks = criteo_df['click'].sum()
    overall_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    
    print(f"Total samples: {total_impressions:,}")
    print(f"Total 'clicks' (high engagement): {total_clicks:,}")
    print(f"Overall CTR: {overall_ctr:.2f}%")
    
    # CTR by channel
    print("\nCTR by Channel:")
    for channel in criteo_df['cat_0'].unique():
        if channel:
            channel_data = criteo_df[criteo_df['cat_0'] == channel]
            channel_ctr = channel_data['click'].mean() * 100
            print(f"  {channel:20} {channel_ctr:6.2f}%  ({len(channel_data):,} samples)")
    
    # CTR by device
    print("\nCTR by Device:")
    for device in criteo_df['cat_3'].unique():
        if device:
            device_data = criteo_df[criteo_df['cat_3'] == device]
            device_ctr = device_data['click'].mean() * 100
            print(f"  {device:20} {device_ctr:6.2f}%  ({len(device_data):,} samples)")
    
    # CTR by hour segment
    print("\nCTR by Time of Day:")
    time_segments = {'0': 'Night (12am-6am)', '1': 'Morning (6am-12pm)', 
                    '2': 'Afternoon (12pm-6pm)', '3': 'Evening (6pm-12am)'}
    for seg in ['0', '1', '2', '3']:
        seg_data = criteo_df[criteo_df['cat_4'] == seg]
        if len(seg_data) > 0:
            seg_ctr = seg_data['click'].mean() * 100
            print(f"  {time_segments[seg]:20} {seg_ctr:6.2f}%  ({len(seg_data):,} samples)")
    
    print(f"\n✅ Real CTR training data ready: {criteo_file}")
    print(f"   Use this to train the Criteo model for realistic CTR predictions!")
    
else:
    print("\n❌ No data collected")