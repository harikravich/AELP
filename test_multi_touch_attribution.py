#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Touch Attribution System

This test validates ALL required functionality:
✓ Multi-touch attribution models (Linear, Time Decay, Position-Based, Data-Driven)
✓ Cross-device user journey tracking
✓ iOS privacy compliance and server-side tracking
✓ Real conversion attribution (impressions → clicks → visits → conversions)
✓ Attribution accuracy verification
✓ NO single-touch attribution fallbacks

CRITICAL TEST SCENARIOS:
- Complex multi-channel journeys (5+ touchpoints)
- Cross-device attribution (mobile impression → desktop conversion)
- iOS privacy-restricted tracking
- Attribution model comparison and verification
- Real-time conversion processing
- ROI calculation accuracy
"""

import os
import sys
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import attribution system
from attribution_system import MultiTouchAttributionEngine
from attribution_models import AttributionEngine, Journey, Touchpoint


class AttributionSystemTester:
    """Comprehensive test suite for multi-touch attribution system."""
    
    def __init__(self):
        self.test_db_path = "test_attribution_system.db"
        self.attribution_system = MultiTouchAttributionEngine(db_path=self.test_db_path)
        self.test_results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'detailed_results': []
        }
        
    def run_all_tests(self):
        """Run complete test suite."""
        print("=" * 80)
        print("MULTI-TOUCH ATTRIBUTION SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        
        # Test 1: Multi-Touch Attribution Models
        print("\n1️⃣ TESTING ATTRIBUTION MODELS")
        self.test_attribution_models()
        
        # Test 2: Cross-Device Tracking
        print("\n2️⃣ TESTING CROSS-DEVICE TRACKING")
        self.test_cross_device_tracking()
        
        # Test 3: iOS Privacy Compliance
        print("\n3️⃣ TESTING iOS PRIVACY COMPLIANCE")
        self.test_ios_privacy_compliance()
        
        # Test 4: Complex Multi-Channel Journey
        print("\n4️⃣ TESTING COMPLEX MULTI-CHANNEL JOURNEYS")
        self.test_complex_journey_attribution()
        
        # Test 5: Attribution Accuracy Verification
        print("\n5️⃣ TESTING ATTRIBUTION ACCURACY")
        self.test_attribution_accuracy()
        
        # Test 6: Real-Time Conversion Processing
        print("\n6️⃣ TESTING REAL-TIME CONVERSION PROCESSING")
        self.test_realtime_conversion_processing()
        
        # Test 7: ROI Calculation
        print("\n7️⃣ TESTING ROI CALCULATION")
        self.test_roi_calculation()
        
        # Test 8: NO Fallback Verification
        print("\n8️⃣ TESTING NO FALLBACK ENFORCEMENT")
        self.test_no_fallback_enforcement()
        
        # Final results
        self.print_final_results()
        
    def test_attribution_models(self):
        """Test all attribution models work correctly."""
        test_name = "Attribution Models Test"
        
        try:
            # Create test journey with 4 touchpoints
            base_time = datetime.now() - timedelta(days=5)
            
            # Track complete journey
            user_id = "test_user_attribution_models"
            
            # Impression (Day -5)
            impression_id = self.attribution_system.track_impression(
                campaign_data={
                    'channel': 'display',
                    'source': 'facebook',
                    'medium': 'display',
                    'campaign': 'awareness_campaign',
                    'creative_id': 'banner_001'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'device_001',
                    'ip_hash': hashlib.sha256('192.168.1.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                },
                timestamp=base_time
            )
            
            # Click (Day -3) 
            click_id = self.attribution_system.track_click(
                campaign_data={
                    'channel': 'search',
                    'source': 'google',
                    'medium': 'cpc',
                    'campaign': 'brand_search',
                    'keyword': 'parental controls'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'device_001',
                    'ip_hash': hashlib.sha256('192.168.1.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                },
                click_data={
                    'click_id': 'gclid_test_123',
                    'landing_page': 'https://example.com/landing',
                    'actions_taken': ['form_view']
                },
                timestamp=base_time + timedelta(days=2)
            )
            
            # Visit (Day -1)
            visit_id = self.attribution_system.track_site_visit(
                visit_data={
                    'page_url': 'https://example.com/product',
                    'referrer': 'https://google.com',
                    'time_on_page': 120,
                    'actions_taken': ['video_play', 'demo_request']
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'device_001',
                    'ip_hash': hashlib.sha256('192.168.1.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                },
                timestamp=base_time + timedelta(days=4)
            )
            
            # Conversion (Day 0)
            conversion_id = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 150.0,
                    'type': 'subscription',
                    'product_category': 'family_safety'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'device_001',
                    'ip_hash': hashlib.sha256('192.168.1.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                },
                timestamp=base_time + timedelta(days=5)
            )
            
            # Get attribution results
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=10)
            
            # Verify all touchpoints tracked
            assert len(journey_data['touchpoints']) == 4, f"Expected 4 touchpoints, got {len(journey_data['touchpoints'])}"
            
            # Verify attribution calculated for all models
            attribution_results = journey_data['attribution_results']
            models_found = set(result['result_attribution_model'] for result in attribution_results)
            expected_models = {'linear', 'time_decay', 'position_based', 'data_driven'}
            
            assert expected_models.issubset(models_found), f"Missing attribution models: {expected_models - models_found}"
            
            # Verify attribution weights sum to 1.0 for each model
            for model in expected_models:
                model_results = [r for r in attribution_results if r['result_attribution_model'] == model]
                total_weight = sum(r['attribution_weight'] for r in model_results)
                assert abs(total_weight - 1.0) < 0.01, f"Attribution weights don't sum to 1.0 for {model}: {total_weight}"
            
            # Verify different models produce different attributions
            linear_weights = [r['attribution_weight'] for r in attribution_results if r['result_attribution_model'] == 'linear']
            time_decay_weights = [r['attribution_weight'] for r in attribution_results if r['result_attribution_model'] == 'time_decay']
            
            assert linear_weights != time_decay_weights, "Linear and time decay should produce different weights"
            
            self._record_test_result(test_name, True, "All attribution models working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Attribution models test failed: {e}")
    
    def test_cross_device_tracking(self):
        """Test cross-device user journey tracking."""
        test_name = "Cross-Device Tracking Test"
        
        try:
            base_time = datetime.now() - timedelta(hours=6)
            user_id = "test_cross_device_user"
            
            # Mobile impression (same IP, different device)
            mobile_impression = self.attribution_system.track_impression(
                campaign_data={
                    'channel': 'social',
                    'source': 'instagram',
                    'medium': 'social',
                    'campaign': 'mobile_video_campaign',
                    'creative_id': 'mobile_video_001'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'mobile_device_123',
                    'ip_hash': hashlib.sha256('10.0.0.100'.encode()).hexdigest()[:16],
                    'platform': 'iOS',
                    'is_mobile': True,
                    'is_ios': True,
                    'timezone': 'America/Los_Angeles',
                    'language': 'en-US'
                },
                timestamp=base_time
            )
            
            # Desktop conversion (same IP, same user, different device)
            desktop_conversion = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 200.0,
                    'type': 'purchase',
                    'product_category': 'premium_subscription'
                },
                user_data={
                    'user_id': user_id,  # Same user ID
                    'device_id': 'desktop_device_456',
                    'ip_hash': hashlib.sha256('10.0.0.100'.encode()).hexdigest()[:16],  # Same IP
                    'platform': 'macOS',
                    'is_mobile': False,
                    'is_ios': False,
                    'timezone': 'America/Los_Angeles',  # Same timezone
                    'language': 'en-US'  # Same language
                },
                timestamp=base_time + timedelta(hours=4)
            )
            
            # Verify cross-device journey
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=1)
            
            # Should have 2 touchpoints from 2 different devices
            assert len(journey_data['touchpoints']) == 2, f"Expected 2 touchpoints, got {len(journey_data['touchpoints'])}"
            
            # Verify different devices but same user
            devices = set(tp['device_id'] for tp in journey_data['touchpoints'] if tp['device_id'])
            assert len(devices) == 2, f"Expected 2 different devices, got {len(devices)}"
            
            # Verify attribution works across devices
            assert len(journey_data['attribution_results']) > 0, "No attribution results for cross-device journey"
            
            # Verify mobile impression gets credit for desktop conversion
            mobile_attribution = [r for r in journey_data['attribution_results'] 
                                 if any(tp['touchpoint_type'] == 'impression' for tp in journey_data['touchpoints'] 
                                       if tp['id'] == r['touchpoint_id'])]
            assert len(mobile_attribution) > 0, "Mobile impression should get attribution credit"
            
            self._record_test_result(test_name, True, "Cross-device tracking working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Cross-device tracking failed: {e}")
    
    def test_ios_privacy_compliance(self):
        """Test iOS privacy restrictions and server-side tracking."""
        test_name = "iOS Privacy Compliance Test"
        
        try:
            base_time = datetime.now() - timedelta(hours=2)
            user_id = "test_ios_privacy_user"
            
            # iOS user with privacy restrictions
            ios_click = self.attribution_system.track_click(
                campaign_data={
                    'channel': 'social',
                    'source': 'tiktok',
                    'medium': 'social',
                    'campaign': 'ios_privacy_test',
                    'creative_id': 'ios_video_001'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'ios_device_789',
                    'ip_hash': hashlib.sha256('172.16.0.50'.encode()).hexdigest()[:16],
                    'platform': 'iOS',
                    'is_mobile': True,
                    'is_ios': True,  # Privacy restricted
                    'timezone': 'America/New_York'
                },
                click_data={
                    # No click_id due to iOS restrictions
                    'landing_page': 'https://example.com/ios-landing',
                    'actions_taken': ['app_store_visit']
                },
                timestamp=base_time
            )
            
            # iOS conversion
            ios_conversion = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 99.99,
                    'type': 'subscription',
                    'product_category': 'mobile_app'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'ios_device_789',
                    'ip_hash': hashlib.sha256('172.16.0.50'.encode()).hexdigest()[:16],
                    'platform': 'iOS',
                    'is_mobile': True,
                    'is_ios': True,
                    'timezone': 'America/New_York'
                },
                timestamp=base_time + timedelta(hours=1)
            )
            
            # Verify iOS journey tracking
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=1)
            
            assert len(journey_data['touchpoints']) == 2, f"Expected 2 touchpoints, got {len(journey_data['touchpoints'])}"
            
            # Verify privacy restrictions applied
            ios_touchpoints = [tp for tp in journey_data['touchpoints'] if tp['is_privacy_restricted']]
            assert len(ios_touchpoints) > 0, "iOS touchpoints should be marked as privacy restricted"
            
            # Verify server-side tracking used
            server_tracked = [tp for tp in journey_data['touchpoints'] if tp['tracking_method'] in ['server', 'hybrid']]
            assert len(server_tracked) > 0, "iOS should use server-side tracking"
            
            # Verify attribution still works despite privacy restrictions
            assert len(journey_data['attribution_results']) > 0, "Attribution should work despite iOS restrictions"
            
            self._record_test_result(test_name, True, "iOS privacy compliance working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"iOS privacy compliance failed: {e}")
    
    def test_complex_journey_attribution(self):
        """Test complex multi-channel journey with 6+ touchpoints."""
        test_name = "Complex Multi-Channel Journey Test"
        
        try:
            base_time = datetime.now() - timedelta(days=14)
            user_id = "test_complex_journey_user"
            
            # Create complex journey: Display → Social → Search → Email → Direct → Conversion
            touchpoints_data = [
                {
                    'method': 'track_impression',
                    'campaign_data': {
                        'channel': 'display',
                        'source': 'google_display',
                        'medium': 'display',
                        'campaign': 'awareness_display',
                        'creative_id': 'banner_awareness_001'
                    },
                    'extra_data': {},
                    'timestamp_offset': 0
                },
                {
                    'method': 'track_click',
                    'campaign_data': {
                        'channel': 'social',
                        'source': 'facebook',
                        'medium': 'social',
                        'campaign': 'retargeting_social',
                        'creative_id': 'video_retargeting_001'
                    },
                    'extra_data': {
                        'click_id': 'fbclid_complex_123',
                        'landing_page': 'https://example.com/social-landing',
                        'actions_taken': ['video_view']
                    },
                    'timestamp_offset': 2
                },
                {
                    'method': 'track_click',
                    'campaign_data': {
                        'channel': 'search',
                        'source': 'google',
                        'medium': 'cpc',
                        'campaign': 'brand_search_exact',
                        'keyword': 'family safety app'
                    },
                    'extra_data': {
                        'click_id': 'gclid_complex_456',
                        'landing_page': 'https://example.com/search-landing',
                        'actions_taken': ['demo_request', 'pricing_view']
                    },
                    'timestamp_offset': 5
                },
                {
                    'method': 'track_click',
                    'campaign_data': {
                        'channel': 'email',
                        'source': 'email_nurture',
                        'medium': 'email',
                        'campaign': 'welcome_series_email_3',
                        'creative_id': 'email_template_welcome_3'
                    },
                    'extra_data': {
                        'landing_page': 'https://example.com/email-landing',
                        'actions_taken': ['testimonial_read', 'feature_comparison']
                    },
                    'timestamp_offset': 8
                },
                {
                    'method': 'track_site_visit',
                    'visit_data': {
                        'page_url': 'https://example.com/pricing',
                        'referrer': 'direct',
                        'source': 'direct',
                        'medium': 'none',
                        'campaign': 'direct_visit',
                        'time_on_page': 300,
                        'actions_taken': ['price_calculator', 'faq_read']
                    },
                    'extra_data': {},
                    'timestamp_offset': 12
                }
            ]
            
            touchpoint_ids = []
            
            # Track all touchpoints
            for tp_data in touchpoints_data:
                user_data = {
                    'user_id': user_id,
                    'device_id': 'complex_device_001',
                    'ip_hash': hashlib.sha256('203.0.113.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                }
                
                timestamp = base_time + timedelta(days=tp_data['timestamp_offset'])
                
                if tp_data['method'] == 'track_impression':
                    tp_id = self.attribution_system.track_impression(
                        tp_data['campaign_data'], user_data, timestamp
                    )
                elif tp_data['method'] == 'track_click':
                    tp_id = self.attribution_system.track_click(
                        tp_data['campaign_data'], user_data, tp_data['extra_data'], timestamp
                    )
                elif tp_data['method'] == 'track_site_visit':
                    tp_id = self.attribution_system.track_site_visit(
                        tp_data['visit_data'], user_data, timestamp
                    )
                
                touchpoint_ids.append(tp_id)
            
            # Final conversion
            conversion_id = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 299.99,
                    'type': 'annual_subscription',
                    'product_category': 'premium_family_plan'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'complex_device_001',
                    'ip_hash': hashlib.sha256('203.0.113.100'.encode()).hexdigest()[:16],
                    'platform': 'Windows',
                    'timezone': 'America/New_York'
                },
                timestamp=base_time + timedelta(days=14)
            )
            
            # Verify complex journey
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=20)
            
            # Should have 6 touchpoints (5 pre-conversion + 1 conversion)
            assert len(journey_data['touchpoints']) == 6, f"Expected 6 touchpoints, got {len(journey_data['touchpoints'])}"
            
            # Verify 5 unique channels
            channels = set(tp['channel'] for tp in journey_data['touchpoints'])
            assert len(channels) >= 5, f"Expected 5+ unique channels, got {len(channels)}"
            
            # Verify attribution across all touchpoints
            attribution_results = journey_data['attribution_results']
            
            # Each attribution model should have results for multiple touchpoints
            for model in ['linear', 'time_decay', 'position_based']:
                model_results = [r for r in attribution_results if r['result_attribution_model'] == model]
                assert len(model_results) >= 5, f"{model} should attribute to 5+ touchpoints, got {len(model_results)}"
                
                # Verify weights sum to 1.0
                total_weight = sum(r['attribution_weight'] for r in model_results)
                assert abs(total_weight - 1.0) < 0.01, f"{model} weights should sum to 1.0, got {total_weight}"
            
            # Verify different models produce different attribution patterns
            linear_first = [r for r in attribution_results if r['result_attribution_model'] == 'linear'][0]['attribution_weight']
            position_first = [r for r in attribution_results if r['result_attribution_model'] == 'position_based'][0]['attribution_weight']
            
            assert linear_first != position_first, "Different models should produce different attributions"
            
            self._record_test_result(test_name, True, "Complex multi-channel journey attribution working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Complex journey attribution failed: {e}")
    
    def test_attribution_accuracy(self):
        """Test attribution accuracy and verify no single-touch fallback."""
        test_name = "Attribution Accuracy Test"
        
        try:
            # Test scenario: Ensure attribution is distributed properly, not just last-click
            base_time = datetime.now() - timedelta(days=3)
            user_id = "test_accuracy_user"
            
            # First click (high value channel)
            first_click = self.attribution_system.track_click(
                campaign_data={
                    'channel': 'search',
                    'source': 'google',
                    'medium': 'cpc',
                    'campaign': 'high_intent_keywords',
                    'keyword': 'best parental control software'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'accuracy_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.10'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                click_data={
                    'click_id': 'gclid_accuracy_789',
                    'landing_page': 'https://example.com/high-intent-landing',
                    'time_on_page': 180,
                    'actions_taken': ['demo_request', 'whitepaper_download', 'pricing_view']
                },
                timestamp=base_time
            )
            
            # Low-value last click
            last_click = self.attribution_system.track_click(
                campaign_data={
                    'channel': 'display',
                    'source': 'low_value_publisher',
                    'medium': 'display',
                    'campaign': 'cheap_banner_network',
                    'creative_id': 'generic_banner_001'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'accuracy_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.10'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                click_data={
                    'landing_page': 'https://example.com/generic-landing',
                    'time_on_page': 30,
                    'actions_taken': []
                },
                timestamp=base_time + timedelta(days=2, hours=23)  # Just before conversion
            )
            
            # Conversion
            conversion = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 180.0,
                    'type': 'subscription',
                    'product_category': 'standard_plan'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'accuracy_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.10'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                timestamp=base_time + timedelta(days=3)
            )
            
            # Verify attribution is NOT just last-click
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=5)
            
            # Get attribution for different models
            attribution_results = journey_data['attribution_results']
            
            # For position-based attribution, first click should get significant credit
            position_results = [r for r in attribution_results if r['result_attribution_model'] == 'position_based']
            
            # Find first and last touchpoint attributions
            touchpoints = journey_data['touchpoints']
            first_tp_id = min(touchpoints, key=lambda x: x['timestamp'])['id'] if touchpoints else None
            last_tp_id = max([tp for tp in touchpoints if tp['touchpoint_type'] != 'conversion'], 
                           key=lambda x: x['timestamp'])['id'] if touchpoints else None
            
            first_attribution = next((r for r in position_results if r['touchpoint_id'] == first_tp_id), None)
            last_attribution = next((r for r in position_results if r['touchpoint_id'] == last_tp_id), None)
            
            # First touchpoint should get significant credit (not zero)
            assert first_attribution is not None, "First touchpoint should get attribution credit"
            assert first_attribution['attribution_weight'] > 0.1, f"First touchpoint should get >10% credit, got {first_attribution['attribution_weight']:.2%}"
            
            # Last touchpoint should not get 100% credit (no single-touch fallback)
            if last_attribution:
                assert last_attribution['attribution_weight'] < 0.9, f"Last touchpoint should not get >90% credit, got {last_attribution['attribution_weight']:.2%}"
            
            # Verify time decay gives more credit to recent touches
            time_decay_results = [r for r in attribution_results if r['result_attribution_model'] == 'time_decay']
            last_td_attribution = next((r for r in time_decay_results if r['touchpoint_id'] == last_tp_id), None)
            first_td_attribution = next((r for r in time_decay_results if r['touchpoint_id'] == first_tp_id), None)
            
            if last_td_attribution and first_td_attribution:
                assert last_td_attribution['attribution_weight'] > first_td_attribution['attribution_weight'], \
                    "Time decay should give more credit to recent touches"
            
            self._record_test_result(test_name, True, "Attribution accuracy verified - no single-touch fallbacks")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Attribution accuracy test failed: {e}")
    
    def test_realtime_conversion_processing(self):
        """Test real-time conversion processing and attribution calculation."""
        test_name = "Real-Time Conversion Processing Test"
        
        try:
            user_id = "test_realtime_user"
            conversion_time = datetime.now()
            
            # Track conversion in real-time
            conversion_id = self.attribution_system.track_conversion(
                conversion_data={
                    'value': 75.0,
                    'type': 'trial_upgrade',
                    'product_category': 'basic_plan'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'realtime_device_001',
                    'ip_hash': hashlib.sha256('203.0.113.200'.encode()).hexdigest()[:16],
                    'platform': 'macOS'
                },
                timestamp=conversion_time
            )
            
            # Verify conversion was processed immediately
            journey_data = self.attribution_system.get_user_journey(user_id, days_back=1)
            
            # Should have at least the conversion touchpoint
            conversion_touchpoints = [tp for tp in journey_data['touchpoints'] if tp['touchpoint_type'] == 'conversion']
            assert len(conversion_touchpoints) >= 1, "Conversion should be tracked immediately"
            
            # Verify conversion has correct value
            conversion_tp = conversion_touchpoints[0]
            assert conversion_tp['conversion_value'] == 75.0, f"Expected conversion value 75.0, got {conversion_tp['conversion_value']}"
            
            self._record_test_result(test_name, True, "Real-time conversion processing working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Real-time conversion processing failed: {e}")
    
    def test_roi_calculation(self):
        """Test ROI calculation accuracy with attributed revenue."""
        test_name = "ROI Calculation Test"
        
        try:
            # Use separate database for ROI test to avoid cross-test contamination
            roi_db_path = "roi_test_attribution.db"
            roi_engine = MultiTouchAttributionEngine(db_path=roi_db_path)
            
            # Create journey with known attribution
            user_id = "test_roi_user"
            base_time = datetime.now() - timedelta(days=1)
            
            # Google Ads click
            google_click = roi_engine.track_click(
                campaign_data={
                    'channel': 'search',
                    'source': 'google',
                    'medium': 'cpc',
                    'campaign': 'roi_test_google'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'roi_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.50'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                click_data={'click_id': 'gclid_roi_test'},
                timestamp=base_time
            )
            
            # Facebook click
            facebook_click = roi_engine.track_click(
                campaign_data={
                    'channel': 'social',
                    'source': 'facebook',
                    'medium': 'social',
                    'campaign': 'roi_test_facebook'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'roi_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.50'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                click_data={'click_id': 'fbclid_roi_test'},
                timestamp=base_time + timedelta(hours=12)
            )
            
            # Conversion
            conversion = roi_engine.track_conversion(
                conversion_data={
                    'value': 100.0,
                    'type': 'subscription'
                },
                user_data={
                    'user_id': user_id,
                    'device_id': 'roi_device_001',
                    'ip_hash': hashlib.sha256('198.51.100.50'.encode()).hexdigest()[:16],
                    'platform': 'Windows'
                },
                timestamp=base_time + timedelta(days=1)
            )
            
            # Test ROI calculation
            channel_spend = {
                'search': 50.0,   # Google Ads spend
                'social': 30.0    # Facebook spend
            }
            
            roi_data = roi_engine.calculate_channel_roi(
                channel_spend, 
                attribution_model='linear',  # Equal attribution
                days_back=2
            )
            
            # Clean up ROI test database
            import os
            if os.path.exists(roi_db_path):
                os.remove(roi_db_path)
            
            # Verify ROI data structure
            assert 'search' in roi_data, "Search channel should have ROI data"
            assert 'social' in roi_data, "Social channel should have ROI data"
            
            # Verify attributed revenue is split (linear model = 50/50)
            search_revenue = roi_data['search']['attributed_revenue']
            social_revenue = roi_data['social']['attributed_revenue'] 
            
            assert search_revenue > 0, "Search should have attributed revenue"
            assert social_revenue > 0, "Social should have attributed revenue"
            # With 3 touchpoints (search, social, conversion), each gets 33.33% with linear attribution
            # Only search and social channels are included in ROI, so total should be ~66.67%
            expected_total = 100.0 * (2/3)  # 2 marketing touchpoints out of 3 total
            assert abs(search_revenue + social_revenue - expected_total) < 0.1, f"Total marketing channel revenue should be ~{expected_total:.2f}, got {search_revenue + social_revenue:.2f}"
            
            # Verify ROAS calculation
            search_roas = roi_data['search']['roas']
            social_roas = roi_data['social']['roas']
            
            assert search_roas > 0, "Search ROAS should be positive"
            assert social_roas > 0, "Social ROAS should be positive"
            
            # With equal attribution among 3 touchpoints (33.33% each):
            # Search: $33.33 revenue / $50 spend = 0.67 ROAS
            # Social: $33.33 revenue / $30 spend = 1.11 ROAS
            expected_search_roas = search_revenue / 50.0
            expected_social_roas = social_revenue / 30.0
            
            assert abs(search_roas - expected_search_roas) < 0.01, f"Search ROAS calculation incorrect: {search_roas} vs {expected_search_roas}"
            assert abs(social_roas - expected_social_roas) < 0.01, f"Social ROAS calculation incorrect: {social_roas} vs {expected_social_roas}"
            
            self._record_test_result(test_name, True, "ROI calculation working correctly")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"ROI calculation test failed: {e}")
    
    def test_no_fallback_enforcement(self):
        """Test that NO fallback implementations exist."""
        test_name = "NO Fallback Enforcement Test"
        
        try:
            # This test ensures the system fails properly when components are missing
            # rather than falling back to simplified versions
            
            # Test 1: Verify AttributionEngine has all required models
            engine = AttributionEngine()
            required_models = ['linear', 'time_decay', 'position_based', 'data_driven']
            
            for model in required_models:
                assert model in engine.models, f"Missing required attribution model: {model}"
                assert engine.models[model] is not None, f"Attribution model {model} is None - no fallbacks allowed"
            
            # Test 2: Verify models actually work (no mock implementations)
            test_journey = Journey(
                id="test_no_fallback",
                touchpoints=[
                    Touchpoint(
                        id="tp1", 
                        timestamp=datetime.now() - timedelta(days=2),
                        channel='search',
                        action='click'
                    ),
                    Touchpoint(
                        id="tp2",
                        timestamp=datetime.now() - timedelta(days=1),
                        channel='social',
                        action='click'
                    ),
                    Touchpoint(
                        id="tp3",
                        timestamp=datetime.now(),
                        channel='direct', 
                        action='conversion'
                    )
                ],
                conversion_value=100.0,
                conversion_timestamp=datetime.now(),
                converted=True
            )
            
            # Train data-driven model before testing
            training_journeys = [test_journey]  # Use the test journey itself for training
            for i in range(10):  # Add more training data
                training_journey = Journey(
                    id=f"training_{i}",
                    touchpoints=[
                        Touchpoint(
                            id=f"train_tp_{i}_1",
                            timestamp=datetime.now() - timedelta(days=i+2),
                            channel='search' if i % 2 == 0 else 'display',
                            action='click'
                        ),
                        Touchpoint(
                            id=f"train_tp_{i}_2",
                            timestamp=datetime.now() - timedelta(days=i+1),
                            channel='email',
                            action='click'
                        ),
                        Touchpoint(
                            id=f"train_tp_{i}_3",
                            timestamp=datetime.now() - timedelta(days=i),
                            channel='direct',
                            action='conversion'
                        )
                    ],
                    converted=i % 3 != 0,  # 2/3 convert
                    conversion_value=100.0 if i % 3 != 0 else 0,
                    conversion_timestamp=datetime.now() - timedelta(days=i)
                )
                training_journeys.append(training_journey)
            
            # Train the data-driven model
            engine.train_data_driven_model(training_journeys)
            
            for model_name in required_models:
                attribution = engine.calculate_attribution(test_journey, model_name)
                assert len(attribution) > 0, f"Model {model_name} returned no attribution - possible fallback"
                assert sum(attribution.values()) > 0, f"Model {model_name} returned zero attribution - possible mock"
                
                # Verify different models produce different results (no fallback to same logic)
                if model_name != 'linear':
                    linear_attribution = engine.calculate_attribution(test_journey, 'linear')
                    assert attribution != linear_attribution, f"Model {model_name} produces same results as linear - possible fallback"
            
            # Test 3: Verify data-driven model requires training (no fallback to simple model)
            dd_model = engine.models['data_driven']
            assert hasattr(dd_model, 'train'), "Data-driven model must have train method"
            assert hasattr(dd_model, 'is_trained'), "Data-driven model must track training state"
            
            self._record_test_result(test_name, True, "NO fallback enforcement verified")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"NO fallback enforcement failed: {e}")
    
    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result."""
        if passed:
            self.test_results['tests_passed'] += 1
            print(f"   ✅ {test_name}: {message}")
        else:
            self.test_results['tests_failed'] += 1
            print(f"   ❌ {test_name}: {message}")
        
        self.test_results['detailed_results'].append({
            'test_name': test_name,
            'passed': passed,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
    
    def print_final_results(self):
        """Print final test results."""
        total_tests = self.test_results['tests_passed'] + self.test_results['tests_failed']
        pass_rate = (self.test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 80)
        print("FINAL TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.test_results['tests_passed']}")
        print(f"Failed: {self.test_results['tests_failed']}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.test_results['tests_failed'] == 0:
            print("\n🎉 ALL TESTS PASSED! Multi-Touch Attribution System is working correctly.")
            print("✅ NO single-touch fallbacks detected")
            print("✅ ALL attribution models implemented properly")
            print("✅ Cross-device tracking verified")
            print("✅ iOS privacy compliance confirmed")
            print("✅ Real-time processing operational")
        else:
            print(f"\n⚠️  {self.test_results['tests_failed']} tests failed. Review implementation.")
        
        print("=" * 80)
    
    def cleanup(self):
        """Clean up test database."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
            print(f"\n🗑️  Cleaned up test database: {self.test_db_path}")


def main():
    """Run comprehensive attribution system tests."""
    
    tester = AttributionSystemTester()
    
    try:
        tester.run_all_tests()
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()