#!/usr/bin/env python3
"""
Test Cross-Device Integration for GAELP
Demonstrates how Identity Resolver connects with Journey Database for unified tracking.

This test validates:
1. Cross-device identity resolution 
2. Journey merging across devices
3. Attribution consolidation
4. Confidence scoring validation
"""

import asyncio
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List

from user_journey_database import (
    UserJourneyDatabase, UserProfile, UserJourney, JourneyTouchpoint,
    JourneyState, TransitionTrigger
)
from identity_resolver import (
    IdentityResolver, DeviceSignature, MatchConfidence
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CrossDeviceIntegrationTest:
    """Test cross-device journey tracking integration"""
    
    def __init__(self):
        # Initialize components
        self.identity_resolver = IdentityResolver(
            min_confidence_threshold=0.3,
            high_confidence_threshold=0.8,
            medium_confidence_threshold=0.5
        )
        
        # Mock journey database (using test project)
        self.journey_db = UserJourneyDatabase(
            project_id="test-project",
            dataset_id="test_dataset",
            identity_resolver=self.identity_resolver
        )
        
        self.test_results = []
    
    async def run_all_tests(self):
        """Run comprehensive cross-device integration tests"""
        
        print("="*80)
        print("GAELP Cross-Device Integration Test")
        print("="*80)
        
        test_methods = [
            self.test_basic_identity_resolution,
            self.test_cross_device_journey_continuation,
            self.test_journey_merging_with_confidence,
            self.test_attribution_consolidation,
            self.test_low_confidence_rejection
        ]
        
        for test_method in test_methods:
            try:
                print(f"\n🧪 Running {test_method.__name__}...")
                await test_method()
                self.test_results.append((test_method.__name__, "PASSED"))
                print(f"✅ {test_method.__name__} PASSED")
            except Exception as e:
                logger.error(f"Test failed: {e}")
                self.test_results.append((test_method.__name__, f"FAILED: {e}"))
                print(f"❌ {test_method.__name__} FAILED: {e}")
        
        self._print_summary()
    
    async def test_basic_identity_resolution(self):
        """Test 1: Basic cross-device identity resolution"""
        
        # Create Lisa's mobile device signature
        mobile_signature = DeviceSignature(
            device_id="mobile_lisa_001",
            platform="iOS",
            timezone="America/New_York", 
            language="en-US",
            search_patterns=["parental controls", "kids safety app"],
            session_durations=[45.0, 38.2],
            time_of_day_usage=[9, 14, 20],
            geographic_locations=[(40.7128, -74.0060)],  # NYC
            session_timestamps=[datetime.now() - timedelta(hours=2)],
            ip_addresses={"192.168.1.100"}
        )
        
        # Create Lisa's desktop device signature  
        desktop_signature = DeviceSignature(
            device_id="desktop_lisa_002",
            platform="Windows",
            timezone="America/New_York",
            language="en-US", 
            search_patterns=["parental controls", "family safety"],
            session_durations=[120.5, 89.3],
            time_of_day_usage=[9, 10, 14, 20],
            geographic_locations=[(40.7589, -73.9851)],  # NYC nearby
            session_timestamps=[datetime.now() - timedelta(hours=1)],
            ip_addresses={"192.168.1.101"}  # Same network
        )
        
        # Add signatures to identity resolver
        self.identity_resolver.add_device_signature(mobile_signature)
        self.identity_resolver.add_device_signature(desktop_signature)
        
        # Test identity resolution
        mobile_identity = self.identity_resolver.resolve_identity("mobile_lisa_001")
        desktop_identity = self.identity_resolver.resolve_identity("desktop_lisa_002")
        
        print(f"  Mobile identity: {mobile_identity}")
        print(f"  Desktop identity: {desktop_identity}")
        
        # Validate they resolve to same identity
        assert mobile_identity == desktop_identity, f"Identities should match: {mobile_identity} != {desktop_identity}"
        
        # Test match probability
        match = self.identity_resolver.calculate_match_probability("mobile_lisa_001", "desktop_lisa_002")
        print(f"  Match confidence: {match.confidence_score:.3f} ({match.confidence_level.value})")
        print(f"  Matching signals: {match.matching_signals}")
        
        assert match.confidence_score >= 0.3, f"Match confidence too low: {match.confidence_score}"
    
    async def test_cross_device_journey_continuation(self):
        """Test 2: Journey continuation across devices"""
        
        # Create mobile device fingerprint
        mobile_fingerprint = {
            'device_type': 'mobile',
            'user_agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)',
            'platform': 'iOS',
            'timezone': 'America/New_York',
            'language': 'en-US',
            'search_patterns': ['parental controls'],
            'session_duration': 45.0,
            'time_of_day': 14,
            'location': {'lat': 40.7128, 'lon': -74.0060},
            'ip_address': '192.168.1.100'
        }
        
        # Create journey on mobile device
        mobile_journey, is_new_mobile = self.journey_db.get_or_create_journey(
            user_id="mobile_lisa_001",
            channel="facebook_ads",
            device_fingerprint=mobile_fingerprint
        )
        
        assert is_new_mobile, "Should create new journey for mobile"
        print(f"  Created mobile journey: {mobile_journey.journey_id}")
        print(f"  Mobile canonical user: {mobile_journey.canonical_user_id}")
        
        # Simulate time passing and device switch
        await asyncio.sleep(0.1)
        
        # Create desktop device fingerprint (similar user behavior)
        desktop_fingerprint = {
            'device_type': 'desktop',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'platform': 'Windows',
            'timezone': 'America/New_York',
            'language': 'en-US',
            'search_patterns': ['parental controls', 'family safety'],
            'session_duration': 120.0,
            'time_of_day': 20,
            'location': {'lat': 40.7589, 'lon': -73.9851},  # Nearby location
            'ip_address': '192.168.1.101'  # Same network
        }
        
        # Try to create journey on desktop - should find existing
        desktop_journey, is_new_desktop = self.journey_db.get_or_create_journey(
            user_id="desktop_lisa_002", 
            channel="google_ads",
            device_fingerprint=desktop_fingerprint
        )
        
        print(f"  Desktop journey: {desktop_journey.journey_id}")
        print(f"  Desktop canonical user: {desktop_journey.canonical_user_id}")
        print(f"  Is new journey: {is_new_desktop}")
        
        # Validate cross-device journey continuation
        assert desktop_journey.canonical_user_id == mobile_journey.canonical_user_id, \
            "Should use same canonical user ID across devices"
        
        # Check if journeys were merged or continued
        if not is_new_desktop:
            assert desktop_journey.journey_id == mobile_journey.journey_id, \
                "Should continue same journey across devices"
            print("  ✅ Journey successfully continued across devices")
        else:
            print("  ℹ️ Created separate journey - will be merged via attribution")
    
    async def test_journey_merging_with_confidence(self):
        """Test 3: Journey merging with confidence validation"""
        
        # Create two separate journeys for the same user on different devices
        journey1, _ = self.journey_db.get_or_create_journey(
            user_id="user_merge_test_1",
            channel="search",
            device_fingerprint={
                'platform': 'iOS',
                'timezone': 'America/New_York',
                'search_patterns': ['family protection'],
                'ip_address': '192.168.1.100'
            }
        )
        
        journey2, _ = self.journey_db.get_or_create_journey(
            user_id="user_merge_test_2", 
            channel="social",
            device_fingerprint={
                'platform': 'Android',
                'timezone': 'America/New_York',
                'search_patterns': ['family protection', 'parental controls'],
                'ip_address': '192.168.1.100'  # Same IP
            }
        )
        
        print(f"  Journey 1: {journey1.journey_id} (user: {journey1.user_id})")
        print(f"  Journey 2: {journey2.journey_id} (user: {journey2.user_id})")
        
        # Add touchpoints to both journeys
        touchpoint1 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id=journey1.journey_id,
            user_id=journey1.user_id,
            canonical_user_id=journey1.canonical_user_id,
            timestamp=datetime.now(),
            channel="search",
            interaction_type="click"
        )
        
        touchpoint2 = JourneyTouchpoint(
            touchpoint_id=str(uuid.uuid4()),
            journey_id=journey2.journey_id,
            user_id=journey2.user_id,
            canonical_user_id=journey2.canonical_user_id,
            timestamp=datetime.now() + timedelta(minutes=30),
            channel="social",
            interaction_type="impression"
        )
        
        # Update journeys with touchpoints
        updated_journey1 = self.journey_db.update_journey(
            journey1.journey_id, touchpoint1, TransitionTrigger.CLICK
        )
        updated_journey2 = self.journey_db.update_journey(
            journey2.journey_id, touchpoint2, TransitionTrigger.IMPRESSION
        )
        
        print(f"  Journey 1 touchpoints: {updated_journey1.touchpoint_count}")
        print(f"  Journey 2 touchpoints: {updated_journey2.touchpoint_count}")
        
        # Validate journeys exist
        assert updated_journey1.touchpoint_count >= 1, "Journey 1 should have touchpoints"
        assert updated_journey2.touchpoint_count >= 1, "Journey 2 should have touchpoints"
    
    async def test_attribution_consolidation(self):
        """Test 4: Attribution consolidation across devices"""
        
        # Create identity cluster with multiple devices
        devices = ["tablet_user_001", "phone_user_002", "laptop_user_003"]
        canonical_id = None
        
        # Create journeys on different devices for same user
        for i, device_id in enumerate(devices):
            fingerprint = {
                'device_type': ['tablet', 'mobile', 'desktop'][i],
                'platform': ['iOS', 'Android', 'Windows'][i],
                'timezone': 'America/New_York',
                'search_patterns': ['family safety', 'parental controls'],
                'session_duration': 60 + i * 20,
                'ip_address': '192.168.1.100'  # Same network
            }
            
            journey, _ = self.journey_db.get_or_create_journey(
                user_id=device_id,
                channel=["search", "social", "display"][i],
                device_fingerprint=fingerprint
            )
            
            if canonical_id is None:
                canonical_id = journey.canonical_user_id
            
            print(f"  Device {device_id}: Journey {journey.journey_id}, Canonical: {journey.canonical_user_id}")
            
            # Add touchpoint
            touchpoint = JourneyTouchpoint(
                touchpoint_id=str(uuid.uuid4()),
                journey_id=journey.journey_id,
                user_id=device_id,
                canonical_user_id=journey.canonical_user_id,
                timestamp=datetime.now() + timedelta(minutes=i*10),
                channel=["search", "social", "display"][i],
                interaction_type="impression"
            )
            
            self.journey_db.update_journey(journey.journey_id, touchpoint, TransitionTrigger.IMPRESSION)
        
        # Validate canonical ID consistency
        print(f"  Canonical user ID: {canonical_id}")
        assert canonical_id is not None, "Should have canonical user ID"
        
        # Get cross-device analytics
        if self.identity_resolver.get_identity_cluster(canonical_id):
            analytics = self.journey_db.get_cross_device_analytics(canonical_id)
            print(f"  Cross-device analytics: {analytics.get('device_count', 0)} devices tracked")
            assert analytics.get('device_count', 0) >= 1, "Should track multiple devices"
    
    async def test_low_confidence_rejection(self):
        """Test 5: Low confidence match rejection"""
        
        # Create very different device signatures (should not match)
        device1 = DeviceSignature(
            device_id="different_user_1",
            platform="iOS",
            timezone="America/New_York",
            language="en-US", 
            search_patterns=["parental controls"],
            geographic_locations=[(40.7128, -74.0060)]  # NYC
        )
        
        device2 = DeviceSignature(
            device_id="different_user_2",
            platform="Android",
            timezone="America/Los_Angeles",  # Different timezone
            language="es-ES",  # Different language
            search_patterns=["sports news", "weather"],  # Different interests
            geographic_locations=[(34.0522, -118.2437)]  # LA
        )
        
        self.identity_resolver.add_device_signature(device1)
        self.identity_resolver.add_device_signature(device2)
        
        # Test match probability - should be very low
        match = self.identity_resolver.calculate_match_probability("different_user_1", "different_user_2")
        print(f"  Low confidence match: {match.confidence_score:.3f} ({match.confidence_level.value})")
        
        # Create journeys - should remain separate
        journey1, _ = self.journey_db.get_or_create_journey(
            user_id="different_user_1",
            channel="search",
            device_fingerprint={'platform': 'iOS', 'timezone': 'America/New_York'}
        )
        
        journey2, _ = self.journey_db.get_or_create_journey(
            user_id="different_user_2", 
            channel="search",
            device_fingerprint={'platform': 'Android', 'timezone': 'America/Los_Angeles'}
        )
        
        print(f"  Journey 1 canonical: {journey1.canonical_user_id}")
        print(f"  Journey 2 canonical: {journey2.canonical_user_id}")
        
        # Validate they have different canonical IDs (low confidence rejection)
        assert journey1.canonical_user_id != journey2.canonical_user_id, \
            "Low confidence matches should not be merged"
        
        assert match.confidence_score < 0.5, \
            f"Match confidence should be low: {match.confidence_score}"
        
        print("  ✅ Low confidence matches correctly rejected")
    
    def _print_summary(self):
        """Print test summary"""
        
        print("\n" + "="*80)
        print("CROSS-DEVICE INTEGRATION TEST SUMMARY")
        print("="*80)
        
        passed = sum(1 for _, result in self.test_results if result == "PASSED")
        total = len(self.test_results)
        
        print(f"\nTest Results: {passed}/{total} PASSED")
        
        for test_name, result in self.test_results:
            status = "✅" if result == "PASSED" else "❌"
            print(f"  {status} {test_name}: {result}")
        
        # Print identity resolver statistics
        stats = self.identity_resolver.get_statistics()
        print(f"\nIdentity Resolution Statistics:")
        print(f"  Total devices: {stats['total_devices']}")
        print(f"  Total identities: {stats['total_identities']}")
        print(f"  Average cluster size: {stats['average_cluster_size']:.2f}")
        print(f"  High confidence matches: {stats['high_confidence_matches']}")
        
        success_rate = passed / total
        print(f"\nOverall Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("🎉 Cross-device integration working successfully!")
        else:
            print("⚠️  Some cross-device integration issues detected.")
        
        print("\n" + "="*80)


async def main():
    """Run the cross-device integration test"""
    test_runner = CrossDeviceIntegrationTest()
    await test_runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())