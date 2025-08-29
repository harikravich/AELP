#!/usr/bin/env python3
"""
Convert GA4 data to Criteo format with realistic CTR definition
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

print("="*60)
print("CONVERTING GA4 DATA TO CRITEO FORMAT")
print("="*60)

# Load raw GA4 data
raw_file = Path("/home/hariravichandran/AELP/data/ga4_real_ctr/ga4_raw_data.csv")
df = pd.read_csv(raw_file)

print(f"\nLoaded {len(df):,} rows of GA4 data")

# Create realistic CTR definition
# In web analytics, a "click" in advertising terms is like high engagement
# We'll use a composite score based on engagement, bounce, and conversions
criteo_data = []

for _, row in df.iterrows():
    # Calculate a click probability based on engagement signals
    # High engagement (>0.7) and low bounce (<0.3) indicates interest
    engagement_score = row['engagement_rate'] if row['engagement_rate'] else 0
    bounce_penalty = 1 - (row['bounce_rate'] if row['bounce_rate'] else 0)
    
    # For training, we need binary labels that map to realistic ad CTRs
    # Real ad CTRs are typically 0.1-5%, with search ads higher than display
    # We'll use a probabilistic model based on engagement signals
    
    # Base click probability by channel (realistic CTR ranges)
    channel = row['channel'] if row['channel'] else 'Direct'
    if channel == 'Paid Search':
        base_ctr = 0.03  # 3% for search ads
    elif channel == 'Organic Search':
        base_ctr = 0.025  # 2.5% (high intent)
    elif channel == 'Email':
        base_ctr = 0.02  # 2% for email
    elif channel == 'Direct':
        base_ctr = 0.015  # 1.5% for direct
    elif channel in ['Paid Social', 'Organic Social']:
        base_ctr = 0.008  # 0.8% for social
    elif channel == 'Display':
        base_ctr = 0.002  # 0.2% for display
    else:
        base_ctr = 0.01  # 1% default
    
    # Adjust based on engagement (multiply by up to 3x for high engagement)
    engagement_multiplier = 1 + (engagement_score * 2)  # 1x to 3x
    
    # Adjust based on conversions (10x more likely if converted)
    if row['conversions'] > 0:
        conversion_multiplier = 10
    else:
        conversion_multiplier = 1
    
    # Calculate final click probability
    click_prob = min(base_ctr * engagement_multiplier * conversion_multiplier, 0.5)
    
    # Random assignment based on probability
    click = np.random.choice([0, 1], p=[1 - click_prob, click_prob])
    
    # Map channel to intent score (based on typical channel performance)
    intent_score = 1.0
    channel = row['channel'] if row['channel'] else 'Direct'
    if channel == 'Organic Search':
        intent_score = 2.5  # High intent - actively searching
    elif channel == 'Paid Search':
        intent_score = 2.2  # High intent - clicked ad
    elif channel == 'Direct':
        intent_score = 1.8  # Brand awareness
    elif channel == 'Email':
        intent_score = 1.6  # Engaged subscriber
    elif channel == 'Referral':
        intent_score = 1.4  # Trusted source
    elif channel in ['Paid Social', 'Organic Social']:
        intent_score = 0.8  # Low intent browsing
    elif channel == 'Display':
        intent_score = 0.5  # Very low intent
    elif channel == 'Unassigned':
        intent_score = 1.0  # Unknown
    
    # Extract time features
    try:
        date_obj = datetime.strptime(row['date'], '%Y%m%d')
        day_of_week = date_obj.weekday()
        month = date_obj.month
        day_name = date_obj.strftime('%A')
        week_num = date_obj.isocalendar()[1] % 4
    except:
        day_of_week = 3  # Default to Wednesday
        month = 6  # Default to June
        day_name = 'Wednesday'
        week_num = 2
    
    # Device scoring
    device_score = 1.0
    device = row['device'] if row['device'] else 'desktop'
    if device == 'mobile':
        device_score = 1.2  # Mobile users more likely to click
    elif device == 'tablet':
        device_score = 1.1
    elif device == 'desktop':
        device_score = 1.0
    
    # Build Criteo feature set (39 features total)
    criteo_row = {
        'click': click,
        
        # Numerical features (13 total)
        'num_0': intent_score,  # Channel intent
        'num_1': row['hour'] / 24.0 if row['hour'] else 0.5,  # Hour normalized
        'num_2': day_of_week / 7.0,  # Day of week normalized
        'num_3': row['pageviews'] / max(1, row['sessions']),  # Pages per session
        'num_4': row['avg_session_duration'] / 300.0 if row['avg_session_duration'] else 0,  # Duration normalized
        'num_5': engagement_score,  # Engagement rate (already 0-1)
        'num_6': bounce_penalty,  # Non-bounce rate (inverted)
        'num_7': device_score,  # Device signal
        'num_8': row['conversions'] / max(1, row['sessions']),  # Session CVR
        'num_9': 1.5 if 22 <= row['hour'] or row['hour'] <= 2 else 1.0,  # Late night bonus
        'num_10': np.log1p(row['sessions']) / 10.0,  # Log session volume
        'num_11': np.log1p(row['users']) / 10.0,  # Log user volume
        'num_12': month / 12.0,  # Month normalized
        
        # Categorical features (26 total)
        'cat_0': channel,
        'cat_1': row['source'][:20] if row['source'] else 'direct',
        'cat_2': row['medium'] if row['medium'] else 'none',
        'cat_3': device,
        'cat_4': str(row['hour'] // 6) if row['hour'] else '2',  # Time segment
        'cat_5': 'behavioral_health',  # Industry
        'cat_6': 'aura',  # Brand
        'cat_7': day_name,  # Day name
        'cat_8': f"week_{week_num}",  # Week of month
        'cat_9': 'mental_health' if engagement_score > 0.7 else 'general',  # Topic interest
        'cat_10': 'high' if row['conversions'] > 0 else 'low',  # Value segment
        'cat_11': 'returning' if row['users'] < row['sessions'] else 'new',  # User type
        'cat_12': 'engaged' if engagement_score > 0.5 else 'passive',  # Engagement level
        'cat_13': '',  # Reserved
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

# Convert to DataFrame
criteo_df = pd.DataFrame(criteo_data)

# Save Criteo format
output_file = Path("/home/hariravichandran/AELP/data/ga4_real_ctr/ga4_criteo_realistic.csv")
criteo_df.to_csv(output_file, index=False)

print(f"\n✅ Saved {len(criteo_df):,} rows to {output_file}")

# Calculate statistics
total_clicks = criteo_df['click'].sum()
ctr = (total_clicks / len(criteo_df) * 100)

print("\n" + "="*60)
print("REALISTIC CTR STATISTICS")
print("="*60)
print(f"Total samples: {len(criteo_df):,}")
print(f"Total 'clicks': {total_clicks:,}")
print(f"Overall CTR: {ctr:.2f}%")

# CTR by channel
print("\nCTR by Channel:")
for channel in criteo_df['cat_0'].value_counts().head(10).index:
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

# CTR by engagement level
print("\nCTR by Engagement Level:")
for level in ['engaged', 'passive']:
    level_data = criteo_df[criteo_df['cat_12'] == level]
    if len(level_data) > 0:
        level_ctr = level_data['click'].mean() * 100
        print(f"  {level:20} {level_ctr:6.2f}%  ({len(level_data):,} samples)")

print(f"\n✅ Realistic CTR training data ready!")
print("   This data maps web engagement to ad-like clicks for training")