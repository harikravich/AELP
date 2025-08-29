#!/usr/bin/env python3
"""
Fetch real CTR data from GA4 for Aura to train the Criteo model
Chunks data by week to avoid token limits
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

print("="*60)
print("FETCHING REAL CTR DATA FROM GA4")
print("="*60)

# Configuration
OUTPUT_DIR = Path("/home/hariravichandran/AELP/data/ga4_real_ctr")
OUTPUT_DIR.mkdir(exist_ok=True)

# We'll fetch data week by week to avoid token limits
START_DATE = datetime(2024, 10, 1)  # 3 months back
END_DATE = datetime(2025, 1, 28)
CHUNK_DAYS = 7  # Weekly chunks

all_data = []
current_date = START_DATE

print(f"\nFetching data from {START_DATE.date()} to {END_DATE.date()}")
print(f"Using {CHUNK_DAYS}-day chunks\n")

# Import the MCP tools
try:
    # We'll use subprocess to call the MCP tools
    import subprocess
    
    while current_date < END_DATE:
        chunk_end = min(current_date + timedelta(days=CHUNK_DAYS), END_DATE)
        
        print(f"Fetching: {current_date.date()} to {chunk_end.date()}...", end=" ")
        
        # Build the MCP command to get event data with CTR metrics
        mcp_command = f"""
import json
from datetime import datetime

# Get page views and clicks
start_date = "{current_date.strftime('%Y-%m-%d')}"
end_date = "{chunk_end.strftime('%Y-%m-%d')}"

try:
    # Try to get conversion and click events
    from mcp__ga4__runReport import runReport
    
    response = runReport(
        startDate=start_date,
        endDate=end_date,
        dimensions=[
            {{"name": "date"}},
            {{"name": "sessionDefaultChannelGroup"}},
            {{"name": "sessionSource"}},
            {{"name": "sessionMedium"}},
            {{"name": "deviceCategory"}},
            {{"name": "eventName"}}
        ],
        metrics=[
            {{"name": "sessions"}},
            {{"name": "totalUsers"}},
            {{"name": "screenPageViews"}},
            {{"name": "eventCount"}}
        ]
    )
    print(json.dumps(response))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        
        try:
            # Execute the command
            result = subprocess.run(
                ["python3", "-c", mcp_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if 'error' not in data:
                    print(f"✅ Got {len(data.get('rows', []))} rows")
                    all_data.extend(data.get('rows', []))
                else:
                    print(f"❌ Error: {data['error']}")
            else:
                print(f"❌ Command failed")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        current_date = chunk_end + timedelta(days=1)

except ImportError:
    print("\n⚠️ MCP tools not available, using direct GA4 API")
    
    # Fallback to direct GA4 API
    from google.analytics.data_v1beta import BetaAnalyticsDataClient
    from google.analytics.data_v1beta.types import (
        RunReportRequest, DateRange, Dimension, Metric
    )
    from google.oauth2 import service_account
    
    GA_PROPERTY_ID = "308028264"
    SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
    
    if not SERVICE_ACCOUNT_FILE.exists():
        alt_path = Path.home() / '.config' / 'gaelp' / 'service-account.json'
        if alt_path.exists():
            SERVICE_ACCOUNT_FILE = alt_path
    
    print(f"Using service account: {SERVICE_ACCOUNT_FILE}")
    
    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=['https://www.googleapis.com/auth/analytics.readonly']
    )
    
    client = BetaAnalyticsDataClient(credentials=credentials)
    
    while current_date < END_DATE:
        chunk_end = min(current_date + timedelta(days=CHUNK_DAYS), END_DATE)
        
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
                Dimension(name="landingPagePlusQueryString"),
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="screenPageViews"),
                Metric(name="engagementRate"),
                Metric(name="conversions"),
                Metric(name="bounceRate")
            ]
        )
        
        try:
            response = client.run_report(request)
            
            # Convert to dict format
            rows = []
            for row in response.rows:
                row_data = {
                    'date': row.dimension_values[0].value,
                    'channel': row.dimension_values[1].value,
                    'source': row.dimension_values[2].value,
                    'medium': row.dimension_values[3].value,
                    'device': row.dimension_values[4].value,
                    'landing_page': row.dimension_values[5].value[:100],  # Truncate long URLs
                    'sessions': int(row.metric_values[0].value),
                    'users': int(row.metric_values[1].value),
                    'pageviews': int(row.metric_values[2].value),
                    'engagement_rate': float(row.metric_values[3].value) if row.metric_values[3].value else 0,
                    'conversions': int(row.metric_values[4].value),
                    'bounce_rate': float(row.metric_values[5].value) if row.metric_values[5].value else 0
                }
                rows.append(row_data)
            
            print(f"✅ Got {len(rows)} rows")
            all_data.extend(rows)
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        current_date = chunk_end + timedelta(days=1)

# Process the data
print(f"\n\nTotal rows collected: {len(all_data)}")

if all_data:
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Calculate CTR-like metrics
    # Engagement rate is like CTR for web pages
    df['ctr_proxy'] = df['engagement_rate'] / 100.0  # Convert percentage to decimal
    
    # Calculate conversion rate
    df['cvr'] = df['conversions'] / df['sessions'].replace(0, 1)
    
    # Save raw data
    raw_file = OUTPUT_DIR / "ga4_raw_data.csv"
    df.to_csv(raw_file, index=False)
    print(f"\n✅ Saved raw data to {raw_file}")
    
    # Create Criteo-compatible format
    print("\nConverting to Criteo format...")
    
    criteo_data = []
    for _, row in df.iterrows():
        # Map GA4 data to Criteo features
        criteo_row = {
            # Click/no-click (1 if engaged, 0 if bounced)
            'click': 1 if row['engagement_rate'] > 50 else 0,
            
            # Numerical features
            'num_0': row['engagement_rate'] / 100.0,  # Engagement intensity
            'num_1': row['pageviews'] / max(1, row['sessions']),  # Pages per session
            'num_2': row['sessions'],  # Session volume
            'num_3': row['users'],  # User volume
            'num_4': row['conversions'],  # Conversion count
            'num_5': row['cvr'],  # Conversion rate
            'num_6': row['bounce_rate'] / 100.0,  # Bounce rate
            'num_7': 1.0 if row['device'] == 'mobile' else 0.5,  # Device signal
            'num_8': hash(row['date']) % 7 / 7.0,  # Day of week proxy
            'num_9': int(row['date'].split('-')[1]) / 12.0,  # Month normalized
            'num_10': 0.0,  # Placeholder
            'num_11': 0.0,  # Placeholder
            'num_12': 0.0,  # Placeholder
            
            # Categorical features
            'cat_0': row['channel'],
            'cat_1': row['source'],
            'cat_2': row['medium'],
            'cat_3': row['device'],
            'cat_4': row['date'][:7],  # Year-month
            'cat_5': 'behavioral_health',  # Industry
            'cat_6': 'aura',  # Brand
            'cat_7': row['landing_page'][:20] if pd.notna(row['landing_page']) else '',
            'cat_8': '',  # Placeholder
            'cat_9': '',
            'cat_10': '',
            'cat_11': '',
            'cat_12': '',
            'cat_13': '',
            'cat_14': '',
            'cat_15': '',
            'cat_16': '',
            'cat_17': '',
            'cat_18': '',
            'cat_19': '',
            'cat_20': '',
            'cat_21': '',
            'cat_22': '',
            'cat_23': '',
            'cat_24': '',
            'cat_25': ''
        }
        criteo_data.append(criteo_row)
    
    # Save Criteo format
    criteo_df = pd.DataFrame(criteo_data)
    criteo_file = OUTPUT_DIR / "ga4_criteo_format.csv"
    criteo_df.to_csv(criteo_file, index=False)
    print(f"✅ Saved Criteo format to {criteo_file}")
    
    # Show statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total sessions: {df['sessions'].sum():,}")
    print(f"Total users: {df['users'].sum():,}")
    print(f"Total conversions: {df['conversions'].sum():,}")
    print(f"Average engagement rate: {df['engagement_rate'].mean():.2f}%")
    print(f"Average bounce rate: {df['bounce_rate'].mean():.2f}%")
    print(f"Overall CVR: {(df['conversions'].sum() / df['sessions'].sum() * 100):.3f}%")
    
    # Channel breakdown
    print("\nBy Channel:")
    channel_stats = df.groupby('channel').agg({
        'sessions': 'sum',
        'conversions': 'sum',
        'engagement_rate': 'mean'
    }).round(2)
    print(channel_stats)
    
    print(f"\n✅ Data ready for training at: {criteo_file}")
    
else:
    print("\n❌ No data collected")