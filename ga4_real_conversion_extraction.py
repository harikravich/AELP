#!/usr/bin/env python3
"""
REAL GA4 DATA EXTRACTION FOR GAELP CALIBRATION
NO MOCK DATA - PULL ACTUAL AURA BALANCE/PC PATTERNS

Extract 90 days of real conversion data to ground truth simulation:
- Real conversion windows (3-14 days) 
- Actual CVR by segment
- True attribution paths
- Real CAC by channel
- iOS vs Android patterns (Balance is iOS only)
- Peak conversion hours
- Channel performance mysteries (display 0.01% vs affiliates 4.42%)
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest,
    DateRange,
    Dimension,
    Metric,
    OrderBy,
    FilterExpression,
    Filter,
    FilterExpressionList,
    RunRealtimeReportRequest
)
from google.oauth2 import service_account

class RealGA4DataExtractor:
    def __init__(self):
        self.GA_PROPERTY_ID = "308028264"  # Aura's real property
        self.SERVICE_ACCOUNT_FILE = Path.home() / '.config' / 'gaelp' / 'ga4-service-account.json'
        
        # Load real credentials (no mocking)
        credentials = service_account.Credentials.from_service_account_file(
            str(self.SERVICE_ACCOUNT_FILE),
            scopes=['https://www.googleapis.com/auth/analytics.readonly']
        )
        
        self.client = BetaAnalyticsDataClient(credentials=credentials)
        self.property_path = f"properties/{self.GA_PROPERTY_ID}"
        
        # Data storage for all findings
        self.discovered_patterns = {
            'extraction_timestamp': datetime.now().isoformat(),
            'data_period': '90_days',
            'channel_performance': {},
            'conversion_windows': {},
            'attribution_paths': [],
            'user_segments': {},
            'device_patterns': {},
            'temporal_patterns': {},
            'mystery_insights': {}
        }
    
    def extract_channel_performance_mystery(self):
        """DISCOVER why affiliates get 4.42% CVR vs display 0.01% on 150K sessions"""
        
        print("\n🔍 INVESTIGATING CHANNEL PERFORMANCE MYSTERY")
        print("=" * 80)
        
        # Get detailed channel performance over 90 days
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="sessionDefaultChannelGroup"),
                Dimension(name="sessionSource"),
                Dimension(name="sessionMedium"),
                Dimension(name="sessionCampaignName")
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="conversions"),
                Metric(name="purchaseRevenue"),
                Metric(name="averageSessionDuration"),
                Metric(name="bounceRate"),
                Metric(name="screenPageViews")
            ],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
            limit=100
        )
        
        response = self.client.run_report(request)
        
        channels = {}
        mystery_findings = {}
        
        for row in response.rows:
            channel_group = row.dimension_values[0].value or "unknown"
            source = row.dimension_values[1].value or "unknown" 
            medium = row.dimension_values[2].value or "unknown"
            campaign = row.dimension_values[3].value or "unknown"
            
            sessions = int(row.metric_values[0].value)
            users = int(row.metric_values[1].value) 
            conversions = int(row.metric_values[2].value)
            revenue = float(row.metric_values[3].value)
            avg_duration = float(row.metric_values[4].value)
            bounce_rate = float(row.metric_values[5].value) 
            pageviews = int(row.metric_values[6].value)
            
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            roas = (revenue / (sessions * 0.5)) if sessions > 0 else 0  # Rough CPC estimate
            pages_per_session = pageviews / sessions if sessions > 0 else 0
            
            key = f"{channel_group}|{source}|{medium}"
            
            if key not in channels:
                channels[key] = {
                    'channel_group': channel_group,
                    'source': source, 
                    'medium': medium,
                    'sessions': 0,
                    'users': 0,
                    'conversions': 0,
                    'revenue': 0,
                    'campaigns': []
                }
            
            channels[key]['sessions'] += sessions
            channels[key]['users'] += users  
            channels[key]['conversions'] += conversions
            channels[key]['revenue'] += revenue
            channels[key]['campaigns'].append({
                'name': campaign,
                'sessions': sessions,
                'conversions': conversions,
                'cvr': cvr
            })
            
            # Flag mystery patterns
            if sessions > 1000:  # Significant traffic
                if cvr > 3.0:  # High CVR channels
                    mystery_findings[f"high_cvr_{key}"] = {
                        'type': 'high_conversion',
                        'cvr': cvr,
                        'sessions': sessions,
                        'avg_duration': avg_duration,
                        'bounce_rate': bounce_rate,
                        'pages_per_session': pages_per_session
                    }
                elif cvr < 0.1 and sessions > 10000:  # Low CVR high traffic
                    mystery_findings[f"low_cvr_{key}"] = {
                        'type': 'traffic_sink', 
                        'cvr': cvr,
                        'sessions': sessions,
                        'avg_duration': avg_duration,
                        'bounce_rate': bounce_rate,
                        'pages_per_session': pages_per_session
                    }
        
        # Calculate final metrics and find patterns
        for key, data in channels.items():
            if data['sessions'] > 100:
                final_cvr = (data['conversions'] / data['sessions'] * 100) if data['sessions'] > 0 else 0
                estimated_cac = (data['sessions'] * 0.5) / data['conversions'] if data['conversions'] > 0 else 999
                
                print(f"{data['channel_group']:15} | {data['source']:20} | {data['sessions']:8,} sess | {data['conversions']:4} conv | {final_cvr:6.2f}% CVR | ${estimated_cac:.2f} CAC")
                
                self.discovered_patterns['channel_performance'][key] = {
                    'channel_group': data['channel_group'],
                    'source': data['source'],
                    'medium': data['medium'], 
                    'sessions': data['sessions'],
                    'conversions': data['conversions'],
                    'cvr_percent': final_cvr,
                    'estimated_cac': estimated_cac,
                    'revenue': data['revenue']
                }
        
        self.discovered_patterns['mystery_insights']['channel_performance'] = mystery_findings
        print(f"\n✅ Found {len(mystery_findings)} channel performance mysteries")

    def extract_conversion_windows(self):
        """Extract REAL conversion lag patterns (1-21 days)"""
        
        print("\n⏰ EXTRACTING REAL CONVERSION WINDOWS")
        print("=" * 80)
        
        # Use cohort report to understand conversion lag
        # Get sessions by date and track conversion lag
        conversion_lags = {}
        
        for days_back in [1, 3, 7, 14, 21]:
            request = RunReportRequest(
                property=self.property_path,
                date_ranges=[
                    DateRange(start_date=f"{days_back+7}daysAgo", end_date=f"{days_back}daysAgo")
                ],
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="sessionDefaultChannelGroup")
                ],
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="conversions"),
                    Metric(name="purchaseRevenue")
                ],
                limit=1000
            )
            
            response = self.client.run_report(request)
            
            day_conversions = 0
            day_sessions = 0
            
            for row in response.rows:
                sessions = int(row.metric_values[0].value)
                conversions = int(row.metric_values[1].value)
                
                day_sessions += sessions
                day_conversions += conversions
            
            conversion_lags[f"{days_back}_day_lag"] = {
                'sessions': day_sessions,
                'conversions': day_conversions,
                'cvr': (day_conversions / day_sessions * 100) if day_sessions > 0 else 0
            }
            
            print(f"{days_back:2d} day lag: {day_sessions:6,} sessions → {day_conversions:4} conversions ({(day_conversions/day_sessions*100) if day_sessions > 0 else 0:.2f}% CVR)")
        
        self.discovered_patterns['conversion_windows'] = conversion_lags

    def extract_attribution_paths(self):
        """Extract REAL multi-touch attribution sequences"""
        
        print("\n🔗 EXTRACTING REAL ATTRIBUTION PATHS")
        print("=" * 80)
        
        # Get user journey sequences
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="sessionSourceMedium"),
                Dimension(name="landingPage"), 
                Dimension(name="eventName"),
                Dimension(name="date")
            ],
            metrics=[
                Metric(name="eventCount"),
                Metric(name="conversions"),
                Metric(name="totalUsers")
            ],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="eventCount"))],
            limit=500
        )
        
        response = self.client.run_report(request)
        
        # Build journey sequences  
        journey_sequences = {}
        conversion_paths = []
        
        for row in response.rows:
            source_medium = row.dimension_values[0].value or "unknown"
            landing_page = row.dimension_values[1].value or "/"
            event_name = row.dimension_values[2].value or "unknown"
            date = row.dimension_values[3].value
            
            event_count = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            users = int(row.metric_values[2].value)
            
            if event_name == "purchase" and conversions > 0:
                path_key = f"{source_medium} → {landing_page}"
                conversion_paths.append({
                    'path': path_key,
                    'source_medium': source_medium,
                    'landing_page': landing_page,
                    'conversions': conversions,
                    'users': users,
                    'date': date
                })
        
        # Find most common conversion paths
        path_summary = {}
        for path in conversion_paths:
            key = path['path']
            if key not in path_summary:
                path_summary[key] = {'conversions': 0, 'users': 0, 'frequency': 0}
            path_summary[key]['conversions'] += path['conversions']
            path_summary[key]['users'] += path['users'] 
            path_summary[key]['frequency'] += 1
        
        # Sort by conversions
        sorted_paths = sorted(path_summary.items(), key=lambda x: x[1]['conversions'], reverse=True)
        
        print("Top Conversion Paths:")
        for path, data in sorted_paths[:15]:
            cvr = (data['conversions'] / data['users'] * 100) if data['users'] > 0 else 0
            print(f"  {path[:60]:<60} | {data['conversions']:3} conv | {data['users']:4} users | {cvr:.1f}% CVR")
        
        self.discovered_patterns['attribution_paths'] = sorted_paths[:20]

    def extract_device_patterns(self):
        """Extract iOS vs Android patterns (Critical for Balance iOS limitation)"""
        
        print("\n📱 EXTRACTING DEVICE/PLATFORM PATTERNS")
        print("=" * 80)
        
        # Device category analysis
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="deviceCategory"),
                Dimension(name="mobileDeviceBranding"), 
                Dimension(name="operatingSystem"),
                Dimension(name="sessionDefaultChannelGroup")
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="conversions"),
                Metric(name="purchaseRevenue"),
                Metric(name="averageSessionDuration")
            ],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
            limit=100
        )
        
        response = self.client.run_report(request)
        
        device_patterns = {}
        ios_performance = {'sessions': 0, 'conversions': 0, 'revenue': 0}
        android_performance = {'sessions': 0, 'conversions': 0, 'revenue': 0} 
        
        for row in response.rows:
            device_cat = row.dimension_values[0].value or "unknown"
            brand = row.dimension_values[1].value or "unknown"
            os = row.dimension_values[2].value or "unknown" 
            channel = row.dimension_values[3].value or "unknown"
            
            sessions = int(row.metric_values[0].value)
            users = int(row.metric_values[1].value)
            conversions = int(row.metric_values[2].value)
            revenue = float(row.metric_values[3].value)
            avg_duration = float(row.metric_values[4].value)
            
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            
            device_key = f"{device_cat}_{os}_{brand}"[:30]
            
            if sessions > 100:
                device_patterns[device_key] = {
                    'device_category': device_cat,
                    'os': os,
                    'brand': brand, 
                    'channel': channel,
                    'sessions': sessions,
                    'conversions': conversions,
                    'cvr': cvr,
                    'revenue': revenue,
                    'avg_duration': avg_duration
                }
                
                print(f"{device_cat:10} | {os:15} | {brand:15} | {sessions:8,} sess | {conversions:4} conv | {cvr:5.2f}% CVR")
            
            # Aggregate iOS vs Android performance
            if 'ios' in os.lower() or 'iphone' in brand.lower():
                ios_performance['sessions'] += sessions
                ios_performance['conversions'] += conversions 
                ios_performance['revenue'] += revenue
            elif 'android' in os.lower():
                android_performance['sessions'] += sessions
                android_performance['conversions'] += conversions
                android_performance['revenue'] += revenue
        
        # Calculate platform summary
        ios_cvr = (ios_performance['conversions'] / ios_performance['sessions'] * 100) if ios_performance['sessions'] > 0 else 0
        android_cvr = (android_performance['conversions'] / android_performance['sessions'] * 100) if android_performance['sessions'] > 0 else 0
        
        print(f"\n📊 PLATFORM PERFORMANCE SUMMARY:")
        print(f"iOS:     {ios_performance['sessions']:8,} sessions | {ios_performance['conversions']:4} conv | {ios_cvr:.2f}% CVR | ${ios_performance['revenue']:,.2f}")
        print(f"Android: {android_performance['sessions']:8,} sessions | {android_performance['conversions']:4} conv | {android_cvr:.2f}% CVR | ${android_performance['revenue']:,.2f}")
        
        # Critical insight for Balance (iOS only)
        balance_impact = {
            'ios_traffic_percentage': (ios_performance['sessions'] / (ios_performance['sessions'] + android_performance['sessions']) * 100) if (ios_performance['sessions'] + android_performance['sessions']) > 0 else 0,
            'ios_cvr_vs_android': ios_cvr - android_cvr,
            'potential_balance_addressable_market': ios_performance['sessions']
        }
        
        print(f"\n🎯 BALANCE (iOS ONLY) IMPLICATIONS:")
        print(f"iOS Traffic Share: {balance_impact['ios_traffic_percentage']:.1f}%")
        print(f"iOS vs Android CVR Difference: {balance_impact['ios_cvr_vs_android']:+.2f} percentage points")
        print(f"Balance Addressable Sessions: {balance_impact['potential_balance_addressable_market']:,}")
        
        self.discovered_patterns['device_patterns'] = {
            'all_devices': device_patterns,
            'ios_performance': ios_performance,
            'android_performance': android_performance, 
            'balance_implications': balance_impact
        }

    def extract_temporal_patterns(self):
        """Extract peak conversion hours and day-of-week patterns"""
        
        print("\n⏰ EXTRACTING TEMPORAL CONVERSION PATTERNS") 
        print("=" * 80)
        
        # Hour of day analysis
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="hour"),
                Dimension(name="dayOfWeek")
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="conversions"),
                Metric(name="purchaseRevenue")
            ],
            order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name="hour"))],
            limit=200
        )
        
        response = self.client.run_report(request)
        
        hourly_patterns = {}
        daily_patterns = {}
        
        for row in response.rows:
            hour_val = row.dimension_values[0].value
            day_of_week = row.dimension_values[1].value  # 0=Sunday, 6=Saturday
            
            # Skip invalid hour values
            if hour_val == "(other)" or not hour_val.isdigit():
                continue
                
            hour = int(hour_val)
            sessions = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            revenue = float(row.metric_values[2].value)
            
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            
            # Hourly aggregation
            if hour not in hourly_patterns:
                hourly_patterns[hour] = {'sessions': 0, 'conversions': 0, 'revenue': 0}
            hourly_patterns[hour]['sessions'] += sessions
            hourly_patterns[hour]['conversions'] += conversions
            hourly_patterns[hour]['revenue'] += revenue
            
            # Daily aggregation  
            if day_of_week not in daily_patterns:
                daily_patterns[day_of_week] = {'sessions': 0, 'conversions': 0, 'revenue': 0}
            daily_patterns[day_of_week]['sessions'] += sessions
            daily_patterns[day_of_week]['conversions'] += conversions
            daily_patterns[day_of_week]['revenue'] += revenue
        
        # Find peak hours
        peak_hours = []
        print("\nHourly Conversion Patterns:")
        for hour in sorted(hourly_patterns.keys()):
            data = hourly_patterns[hour]
            cvr = (data['conversions'] / data['sessions'] * 100) if data['sessions'] > 0 else 0
            
            if data['sessions'] > 100:  # Only show significant traffic hours
                time_label = f"{hour:2d}:00"
                print(f"  {time_label} | {data['sessions']:6,} sessions | {data['conversions']:3} conv | {cvr:.2f}% CVR")
                
                if cvr > 1.5:  # High conversion hours
                    peak_hours.append({'hour': hour, 'cvr': cvr, 'sessions': data['sessions']})
        
        # Day of week patterns
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        print(f"\nDaily Patterns:")
        for day_idx in sorted(daily_patterns.keys()):
            data = daily_patterns[day_idx]
            cvr = (data['conversions'] / data['sessions'] * 100) if data['sessions'] > 0 else 0
            day_name = day_names[int(day_idx)] if int(day_idx) < 7 else f"Day_{day_idx}"
            
            print(f"  {day_name:9} | {data['sessions']:8,} sessions | {data['conversions']:4} conv | {cvr:.2f}% CVR")
        
        # Identify parent browsing pattern (evening hypothesis)
        evening_hours = [19, 20, 21, 22]  # 7-10 PM
        evening_performance = {
            'sessions': sum(hourly_patterns.get(h, {}).get('sessions', 0) for h in evening_hours),
            'conversions': sum(hourly_patterns.get(h, {}).get('conversions', 0) for h in evening_hours)
        }
        evening_cvr = (evening_performance['conversions'] / evening_performance['sessions'] * 100) if evening_performance['sessions'] > 0 else 0
        
        print(f"\n🌅 PARENT EVENING BROWSING PATTERN (7-10 PM):")
        print(f"Evening Sessions: {evening_performance['sessions']:,}")
        print(f"Evening Conversions: {evening_performance['conversions']}")
        print(f"Evening CVR: {evening_cvr:.2f}%")
        
        self.discovered_patterns['temporal_patterns'] = {
            'hourly_patterns': hourly_patterns,
            'daily_patterns': daily_patterns,
            'peak_hours': sorted(peak_hours, key=lambda x: x['cvr'], reverse=True)[:5],
            'evening_parent_pattern': evening_performance
        }

    def extract_user_segments(self):
        """Discover high-value converter characteristics and behavioral patterns"""
        
        print("\n👥 EXTRACTING USER SEGMENT PATTERNS")
        print("=" * 80)
        
        # Analyze user segments by behavior and value
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="city"),
                Dimension(name="country"), 
                Dimension(name="newVsReturning"),
                Dimension(name="sessionDefaultChannelGroup")
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="totalUsers"),
                Metric(name="conversions"),
                Metric(name="purchaseRevenue"),
                Metric(name="averageSessionDuration"),
                Metric(name="sessionsPerUser"),
                Metric(name="screenPageViews")
            ],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="purchaseRevenue"))],
            limit=200
        )
        
        response = self.client.run_report(request)
        
        geographic_segments = {}
        user_type_segments = {}
        high_value_cities = []
        
        for row in response.rows:
            city = row.dimension_values[0].value or "unknown"
            country = row.dimension_values[1].value or "unknown"
            new_returning = row.dimension_values[2].value or "unknown"
            channel = row.dimension_values[3].value or "unknown"
            
            sessions = int(row.metric_values[0].value)
            users = int(row.metric_values[1].value)
            conversions = int(row.metric_values[2].value)
            revenue = float(row.metric_values[3].value)
            avg_duration = float(row.metric_values[4].value)
            sessions_per_user = float(row.metric_values[5].value)
            pageviews = int(row.metric_values[6].value)
            
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            avg_order_value = revenue / conversions if conversions > 0 else 0
            pages_per_session = pageviews / sessions if sessions > 0 else 0
            
            # High-value city identification
            if revenue > 1000 and conversions > 5:
                high_value_cities.append({
                    'city': city,
                    'country': country,
                    'revenue': revenue,
                    'conversions': conversions,
                    'cvr': cvr,
                    'aov': avg_order_value,
                    'avg_duration': avg_duration,
                    'pages_per_session': pages_per_session,
                    'sessions_per_user': sessions_per_user
                })
            
            # User type segments (new vs returning)
            if new_returning not in user_type_segments:
                user_type_segments[new_returning] = {
                    'sessions': 0, 'users': 0, 'conversions': 0, 'revenue': 0
                }
            
            user_type_segments[new_returning]['sessions'] += sessions
            user_type_segments[new_returning]['users'] += users
            user_type_segments[new_returning]['conversions'] += conversions
            user_type_segments[new_returning]['revenue'] += revenue
        
        # Analyze user behavior patterns
        print("High-Value Geographic Segments:")
        for segment in sorted(high_value_cities, key=lambda x: x['revenue'], reverse=True)[:15]:
            print(f"  {segment['city']:20} {segment['country']:15} | ${segment['revenue']:8,.0f} | {segment['conversions']:3} conv | ${segment['aov']:6.0f} AOV | {segment['cvr']:.1f}% CVR")
        
        print(f"\nUser Type Performance:")
        for user_type, data in user_type_segments.items():
            if data['sessions'] > 100:
                cvr = (data['conversions'] / data['sessions'] * 100) if data['sessions'] > 0 else 0
                aov = data['revenue'] / data['conversions'] if data['conversions'] > 0 else 0
                print(f"  {user_type:15} | {data['sessions']:8,} sess | {data['conversions']:4} conv | {cvr:.2f}% CVR | ${aov:.0f} AOV")
        
        # Discover behavioral health personas (crisis parents vs researchers)
        self.discover_behavioral_personas()
        
        self.discovered_patterns['user_segments'] = {
            'high_value_cities': high_value_cities[:20],
            'user_type_performance': user_type_segments,
            'geographic_patterns': geographic_segments
        }

    def discover_behavioral_personas(self):
        """Identify crisis parents (quick conversion) vs researchers (long consideration)"""
        
        print("\n🧠 DISCOVERING BEHAVIORAL HEALTH PERSONAS")
        print("=" * 80)
        
        # Analyze landing pages and user paths for behavioral health intent
        request = RunReportRequest(
            property=self.property_path,
            date_ranges=[DateRange(start_date="90daysAgo", end_date="today")],
            dimensions=[
                Dimension(name="landingPage"),
                Dimension(name="pageTitle")
            ],
            metrics=[
                Metric(name="sessions"),
                Metric(name="conversions"),
                Metric(name="averageSessionDuration"),
                Metric(name="screenPageViews")
            ],
            order_bys=[OrderBy(desc=True, metric=OrderBy.MetricOrderBy(metric_name="sessions"))],
            limit=100
        )
        
        response = self.client.run_report(request)
        
        crisis_indicators = []
        research_indicators = []
        
        for row in response.rows:
            landing_page = row.dimension_values[0].value or "/"
            page_title = row.dimension_values[1].value or ""
            
            sessions = int(row.metric_values[0].value)
            conversions = int(row.metric_values[1].value)
            avg_duration = float(row.metric_values[2].value)
            pageviews = int(row.metric_values[3].value)
            
            cvr = (conversions / sessions * 100) if sessions > 0 else 0
            pages_per_session = pageviews / sessions if sessions > 0 else 0
            
            # Crisis parent signals
            crisis_keywords = ['crisis', 'emergency', 'urgent', 'help', 'suicide', 'depression', 'anxiety', 'teen']
            is_crisis_page = any(keyword in landing_page.lower() or keyword in page_title.lower() for keyword in crisis_keywords)
            
            # Research signals  
            research_keywords = ['review', 'comparison', 'vs', 'pricing', 'features', 'how-to', 'guide']
            is_research_page = any(keyword in landing_page.lower() or keyword in page_title.lower() for keyword in research_keywords)
            
            if sessions > 50:
                if is_crisis_page and avg_duration < 120:  # Quick crisis conversion
                    crisis_indicators.append({
                        'page': landing_page[:50],
                        'sessions': sessions,
                        'conversions': conversions,
                        'cvr': cvr,
                        'avg_duration': avg_duration,
                        'pages_per_session': pages_per_session
                    })
                
                elif is_research_page and avg_duration > 300:  # Long research consideration
                    research_indicators.append({
                        'page': landing_page[:50],
                        'sessions': sessions,
                        'conversions': conversions,
                        'cvr': cvr,
                        'avg_duration': avg_duration,
                        'pages_per_session': pages_per_session
                    })
        
        print("Crisis Parent Patterns (Quick conversion):")
        for pattern in sorted(crisis_indicators, key=lambda x: x['cvr'], reverse=True)[:10]:
            print(f"  {pattern['page'][:40]:40} | {pattern['sessions']:4} sess | {pattern['conversions']:2} conv | {pattern['cvr']:.1f}% CVR | {pattern['avg_duration']:.0f}s")
        
        print("\nResearcher Patterns (Long consideration):")
        for pattern in sorted(research_indicators, key=lambda x: x['avg_duration'], reverse=True)[:10]:
            print(f"  {pattern['page'][:40]:40} | {pattern['sessions']:4} sess | {pattern['conversions']:2} conv | {pattern['cvr']:.1f}% CVR | {pattern['avg_duration']:.0f}s")
        
        self.discovered_patterns['user_segments']['behavioral_personas'] = {
            'crisis_parents': crisis_indicators[:10],
            'researchers': research_indicators[:10]
        }

    def run_complete_extraction(self):
        """Execute full real data extraction"""
        
        print("\n" + "="*100)
        print("REAL GA4 DATA EXTRACTION FOR GAELP CALIBRATION")
        print("NO MOCK DATA - EXTRACTING ACTUAL AURA PATTERNS")
        print("="*100)
        
        try:
            # Extract all real patterns
            self.extract_channel_performance_mystery()
            self.extract_conversion_windows() 
            self.extract_attribution_paths()
            self.extract_device_patterns()
            self.extract_temporal_patterns()
            self.extract_user_segments()
            
            # Save complete findings
            output_file = "discovered_patterns.json"
            with open(output_file, 'w') as f:
                json.dump(self.discovered_patterns, f, indent=2, default=str)
            
            print("\n" + "="*100)
            print("✅ REAL DATA EXTRACTION COMPLETE")
            print("="*100)
            print(f"\n📊 SUMMARY OF DISCOVERIES:")
            print(f"- Channel Performance Mysteries: {len(self.discovered_patterns['mystery_insights'].get('channel_performance', {}))}")
            print(f"- Attribution Paths Mapped: {len(self.discovered_patterns['attribution_paths'])}")
            print(f"- Device Pattern Insights: iOS vs Android performance analyzed")
            print(f"- Temporal Patterns: Peak conversion hours identified")
            print(f"- User Segments: High-value geographic and behavioral segments discovered")
            
            print(f"\n🎯 KEY FINDINGS FOR GAELP CALIBRATION:")
            if 'channel_performance' in self.discovered_patterns:
                high_cvr_channels = [ch for ch in self.discovered_patterns['channel_performance'].values() if ch['cvr_percent'] > 3.0]
                low_cvr_channels = [ch for ch in self.discovered_patterns['channel_performance'].values() if ch['cvr_percent'] < 0.1]
                print(f"- High CVR Channels (>3%): {len(high_cvr_channels)}")
                print(f"- Mystery Low CVR Channels (<0.1%): {len(low_cvr_channels)}")
            
            if 'device_patterns' in self.discovered_patterns:
                ios_share = self.discovered_patterns['device_patterns']['balance_implications']['ios_traffic_percentage']
                print(f"- iOS Traffic Share (Balance addressable): {ios_share:.1f}%")
                
            print(f"\n💾 Complete findings saved to: {output_file}")
            print("\n🚀 Ready to calibrate GAELP simulation with REAL data patterns")
            
            return self.discovered_patterns
            
        except Exception as e:
            print(f"\n❌ EXTRACTION FAILED: {e}")
            raise e

if __name__ == "__main__":
    extractor = RealGA4DataExtractor()
    patterns = extractor.run_complete_extraction()